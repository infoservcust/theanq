import os
import re
from typing import Optional

import requests

# The scheme you pick is the book. One year is the default window.
# Figures describe that window. They are not a forecast.

BENCH = {
    "nifty50": (120716, "UTI Nifty 50 Index Fund — Direct Growth"),
    "next50": (143341, "UTI Nifty Next 50 Index Fund — Direct Growth"),
    "nifty500": (147625, "Motilal Oswal Nifty 500 Index Fund — Direct Growth"),
    "mid": (151724, "HDFC Nifty Midcap 150 Index Fund — Direct Growth"),
    "small": (147623, "Motilal Oswal Nifty Smallcap 250 Index Fund — Direct Growth"),
}

ROOMS = [
    "Overview",
    "Performance",
    "Benchmark",
    "Drivers",
    "Risk",
    "Holdings",
    "Crowding",
    "Peers",
    "What-if",
    "Portfolio",
]

# Synthetic demo. Not a disclosure. Names are real; weights are made up.
HOLDINGS = {
    122639: [
        ("Reliance Industries", "INE002A01018", 0.08),
        ("HDFC Bank", "INE040A01034", 0.07),
        ("Infosys", "INE009A01021", 0.06),
        ("ICICI Bank", "INE090A01021", 0.05),
        ("Tata Consultancy Services", "INE467B01029", 0.04),
        ("Alphabet", "US02079K3059", 0.04),
        ("ITC", "INE154A01025", 0.03),
        ("Larsen & Toubro", "INE018A01030", 0.03),
    ],
    120716: [
        ("HDFC Bank", "INE040A01034", 0.12),
        ("Reliance Industries", "INE002A01018", 0.09),
        ("ICICI Bank", "INE090A01021", 0.08),
        ("Infosys", "INE009A01021", 0.06),
        ("Tata Consultancy Services", "INE467B01029", 0.05),
        ("State Bank of India", "INE062A01020", 0.04),
        ("Bharti Airtel", "INE397D01024", 0.04),
        ("Hindustan Unilever", "INE030A01027", 0.03),
    ],
}
HOLDINGS_AS_OF = "2026-09-30"
HOLDINGS_LABEL = "Synthetic demo. Not a fund disclosure. Names are real securities; weights are made up."
DUMMY_LOTS = [
    (122639, "Parag Parikh Flexi Cap Fund - Direct Plan - Growth", 100000, "2024-04-01"),
    (120716, "UTI Nifty 50 Index Fund - Direct Plan - Growth", 100000, "2024-04-01"),
    (143341, "UTI Nifty Next 50 Index Fund - Direct Plan - Growth", 50000, "2024-06-03"),
    (118955, "HDFC Flexi Cap Fund - Direct Plan - Growth Option", 75000, "2024-06-03"),
    (119091, "HDFC Liquid Fund - Direct Plan - Growth Option", 50000, "2025-01-02"),
]

_SCHEME_CACHE: dict[int, dict] = {}


def _get(url: str) -> object:
    response = requests.get(url, timeout=40, headers={"accept": "application/json"})
    if response.status_code != 200:
        raise RuntimeError(f"The fund feed returned {response.status_code}.")
    return response.json()


def load_scheme(code: int) -> dict:
    cached = _SCHEME_CACHE.get(code)
    if cached is not None:
        return cached
    payload = _get(f"https://api.mfapi.in/mf/{code}")
    if not isinstance(payload, dict):
        raise RuntimeError("The fund feed returned an unexpected reply.")
    _SCHEME_CACHE[code] = payload
    return payload


def search_schemes(query: str) -> list[dict]:
    rows = _get(f"https://api.mfapi.in/mf/search?q={requests.utils.quote(query)}")
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        name = str(row.get("schemeName") or "")
        if "IDCW" in name or "Dividend" in name:
            continue
        out.append({"code": int(row["schemeCode"]), "name": name})
        if len(out) == 8:
            break
    return out


def month_end_navs(rows: list) -> list[tuple[str, float]]:
    by_month: dict[str, tuple[int, float]] = {}
    for row in rows or []:
        parts = str(row.get("date", "")).split("-")
        if len(parts) != 3:
            continue
        try:
            day, month, year = (int(parts[0]), int(parts[1]), int(parts[2]))
            nav = float(row["nav"])
        except (TypeError, ValueError):
            continue
        if nav <= 0:
            continue
        ym = f"{year}-{month:02d}"
        time = year * 10000 + month * 100 + day
        prev = by_month.get(ym)
        if prev is None or time > prev[0]:
            by_month[ym] = (time, nav)
    return [(ym, by_month[ym][1]) for ym in sorted(by_month)]


def monthly_returns(navs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    out = []
    for i in range(1, len(navs)):
        prev = navs[i - 1][1]
        if prev <= 0:
            continue
        out.append((navs[i][0], navs[i][1] / prev - 1))
    return out


def align(fund: list[tuple[str, float]], bench: list[tuple[str, float]], months: int):
    bench_by = dict(bench)
    paired = [(ym, ret, bench_by[ym]) for ym, ret in fund if ym in bench_by]
    if len(paired) <= months:
        return paired
    return paired[-months:]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def ols(y: list[float], x: list[float]) -> tuple[float, float, float]:
    n = min(len(y), len(x))
    if n < 3:
        return 0.0, 0.0, 0.0
    xs, ys = x[:n], y[:n]
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((xs[i] - mx) ** 2 for i in range(n))
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    beta = 0.0 if sxx == 0 else sxy / sxx
    alpha = my - beta * mx
    sse = sum((ys[i] - (alpha + beta * xs[i])) ** 2 for i in range(n))
    sst = sum((ys[i] - my) ** 2 for i in range(n))
    r2 = 0.0 if sst == 0 else 1 - sse / sst
    return alpha, beta, r2


def compounded(returns: list[float]) -> float:
    total = 1.0
    for value in returns:
        total *= 1 + value
    return total


def annualised(total: float, months: int) -> float:
    if months <= 0 or total <= 0:
        return 0.0
    return total ** (12 / months) - 1


def index_stats(points: list[tuple[str, float, float]]) -> Optional[dict]:
    if len(points) < 6:
        return None
    y = [p[1] for p in points]
    x = [p[2] for p in points]
    _alpha, beta, r2 = ols(y, x)
    active_m = [p[1] - p[2] for p in points]
    te = _stdev(active_m) * (12 ** 0.5)
    active = annualised(compounded(y) / compounded(x), len(points))
    up = [p for p in points if p[2] > 0]
    down = [p for p in points if p[2] < 0]

    def capture(rows):
        if len(rows) < 2:
            return None
        b = compounded([p[2] for p in rows]) - 1
        if abs(b) < 1e-8:
            return None
        return (compounded([p[1] for p in rows]) - 1) / b

    def side_beta(rows):
        if len(rows) < 4:
            return None
        return ols([p[1] for p in rows], [p[2] for p in rows])[1]

    return {
        "beta": beta,
        "r2": r2,
        "te": te,
        "active": active,
        "ir": 0.0 if te == 0 else active / te,
        "up_capture": capture(up),
        "down_capture": capture(down),
        "up_months": len(up),
        "down_months": len(down),
        "down_beta": side_beta(down),
    }


def drawdown(returns: list[tuple[str, float]]) -> Optional[dict]:
    if len(returns) < 2:
        return None
    wealth = 1.0
    peak = 1.0
    peak_ym = returns[0][0]
    max_dd = 0.0
    worst_peak_ym = peak_ym
    trough_ym = peak_ym
    trough_i = 0
    worst_peak_i = 0
    peak_i = 0
    path = []
    for i, (ym, ret) in enumerate(returns):
        wealth *= 1 + ret
        path.append(wealth)
        if wealth >= peak:
            peak = wealth
            peak_ym = ym
            peak_i = i
        dd = wealth / peak - 1
        if dd < max_dd:
            max_dd = dd
            worst_peak_ym = peak_ym
            worst_peak_i = peak_i
            trough_ym = ym
            trough_i = i
    peak_at = path[worst_peak_i]
    recovered = False
    recovery = None
    for i in range(trough_i + 1, len(path)):
        if path[i] >= peak_at:
            recovered = True
            recovery = i - trough_i
            break
    return {
        "max_dd": max_dd,
        "peak_ym": worst_peak_ym,
        "trough_ym": trough_ym,
        "recovery_months": recovery,
        "recovered": recovered,
    }


def _label(text: str, phrase: str) -> bool:
    return re.search(rf"{re.escape(phrase)}(?!\d)", text) is not None


def yardstick(category: str, name: str, code: int):
    text = f"{category} {name}".lower()
    pick = BENCH["nifty500"]
    note = (
        "The yardstick is an index fund that tracks Nifty 500, not the official total-return index. "
        "Its own small fee is inside the gap."
    )
    if _label(text, "next 50"):
        pick = BENCH["next50"]
        note = "The yardstick is an index fund that tracks the Nifty Next 50, not the official total-return index."
    elif "small cap" in text or "smallcap" in text:
        pick = BENCH["small"]
        note = "The yardstick is an index fund that tracks the Nifty Smallcap 250, not the official total-return index."
    elif "mid cap" in text or "midcap" in text:
        pick = BENCH["mid"]
        note = "The yardstick is an index fund that tracks the Nifty Midcap 150, not the official total-return index."
    elif _label(text, "nifty 50") or "large cap" in text:
        pick = BENCH["nifty50"]
        note = (
            "Large-cap funds are usually measured against the Nifty 100. This feed has a Nifty 50 index fund, so that is the stand-in, and it is not the official total-return index."
            if "large cap" in text and not _label(text, "nifty 50")
            else "The yardstick is an index fund that tracks the Nifty 50, not the official total-return index."
        )
    if pick[0] == code:
        return None, "This is the index fund used as a yardstick for other funds. It is not compared with itself."
    return pick, note


def measurable(category: str, name: str) -> bool:
    text = f"{category} {name}".lower()
    if any(word in text for word in ("debt", "liquid", "overnight", "gilt", "money market")):
        return False
    return "equity" in text or "index" in text or "elss" in text


def plan_of(name: str) -> str:
    if "Direct" in name:
        return "direct"
    if "Regular" in name:
        return "regular"
    return "unknown"


def growth_series(points, has_bench: bool):
    series = []
    fund = 1.0
    bench = 1.0
    for row in points:
        fund *= 1 + row[1]
        if has_bench:
            bench *= 1 + row[2]
        series.append({"ym": row[0], "fund": fund, "bench": bench})
    return series


def analyse(code: int, months: int) -> dict:
    payload = load_scheme(code)
    meta = payload.get("meta") or {}
    name = str(meta.get("scheme_name") or "")
    category = str(meta.get("scheme_category") or "")
    fund_monthly = monthly_returns(month_end_navs(payload.get("data") or []))
    equity = measurable(category, name)
    bench, note = yardstick(category, name, code) if equity else (
        None,
        "This is not an equity fund. The equity index test is left off rather than forced.",
    )
    points = []
    if bench is not None:
        bench_payload = load_scheme(bench[0])
        bench_monthly = monthly_returns(month_end_navs(bench_payload.get("data") or []))
        points = align(fund_monthly, bench_monthly, months)
    fund_only = fund_monthly[-months:]
    used = points if len(points) >= 6 else []
    chosen = used if used else [(ym, ret, 0.0) for ym, ret in fund_only]
    returns = [row[1] for row in chosen]
    stats = index_stats(used) if used else None
    dd = drawdown([(row[0], row[1]) for row in chosen])
    sibling = find_sibling(name)
    return {
        "fund": {
            "code": code,
            "name": name,
            "house": str(meta.get("fund_house") or ""),
            "category": category,
            "isin": meta.get("isin_growth") or None,
            "plan": plan_of(name),
            "idcw": "IDCW" in name or "Dividend" in name,
        },
        "months": months,
        "paired": len(chosen),
        "start": chosen[0][0] if chosen else "",
        "end": chosen[-1][0] if chosen else "",
        "equity": equity,
        "yardstick": None if bench is None else {"code": bench[0], "name": bench[1]},
        "yardstick_note": note,
        "index": stats,
        "fund_return": annualised(compounded(returns), len(returns)),
        "vol": _stdev(returns) * (12 ** 0.5),
        "drawdown": dd,
        "series": growth_series(chosen, bool(used)),
        "sibling": sibling,
        "sample_growth": 100000 * compounded(returns),
        "asked_months": months,
        "returns": returns,
    }


def find_sibling(name: str) -> Optional[dict]:
    want = "Regular" if "Direct" in name else "Direct" if "Regular" in name else ""
    if not want:
        return None
    stem = name.split(" - ")[0]
    try:
        hits = search_schemes(stem)
    except Exception:
        return None
    for hit in hits:
        other = hit["name"]
        if want in other and "Growth" in other and "IDCW" not in other:
            return hit
    return None


def peers_for(analysis: dict) -> list[dict]:
    if not analysis["equity"]:
        return []
    category = analysis["fund"]["category"]
    tail = category.split(" - ")[-1] if category else "equity"
    try:
        hits = search_schemes(f"{tail} direct growth")
    except Exception:
        return []
    rows = []
    months = analysis["months"]
    bench_code = analysis["yardstick"]["code"] if analysis["yardstick"] else None
    bench_monthly = None
    if bench_code is not None:
        bench_monthly = monthly_returns(month_end_navs(load_scheme(bench_code).get("data") or []))
    for hit in hits:
        if hit["code"] == analysis["fund"]["code"]:
            continue
        if "Direct" not in hit["name"] or "Growth" not in hit["name"]:
            continue
        try:
            monthly = monthly_returns(month_end_navs(load_scheme(hit["code"]).get("data") or []))
            if bench_monthly is not None:
                aligned = align(monthly, bench_monthly, months)
                chosen = aligned if aligned else []
            else:
                chosen = [(ym, ret, 0.0) for ym, ret in monthly[-months:]]
            if not chosen:
                continue
            rets = [row[1] for row in chosen]
            dd = drawdown([(row[0], row[1]) for row in chosen])
            short = hit["name"].replace(" - Direct Plan - Growth", "").replace(" - Direct Growth", "")
            rows.append(
                {
                    "name": short,
                    "ann_return": annualised(compounded(rets), len(rets)),
                    "vol": _stdev(rets) * (12 ** 0.5),
                    "max_dd": None if dd is None else dd["max_dd"],
                }
            )
        except Exception:
            continue
        if len(rows) == 5:
            break
    return rows


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def inr(value: float) -> str:
    sign = "-" if value < 0 else ""
    whole = int(round(abs(value)))
    text = str(whole)
    if len(text) <= 3:
        return f"{sign}₹{text}"
    head, tail = text[:-3], text[-3:]
    parts = []
    while head:
        parts.append(head[-2:])
        head = head[:-2]
    return sign + "₹" + ",".join(reversed(parts)) + "," + tail


def verdict(analysis: dict) -> str:
    fund = analysis["fund"]
    if fund["idcw"]:
        return "This plan pays out. Payouts break the return, so the test is not trustworthy until you pick the growth plan."
    if not analysis["equity"]:
        return "This is not an equity fund. The equity index test is off. What you can see is how far it fell, and how long that lasted."
    stats = analysis["index"]
    if not stats:
        return analysis["yardstick_note"]
    if stats["r2"] >= 0.9 and stats["te"] <= 0.03:
        sentence = f"Over this window it stayed close to the yardstick. Tracking error {pct(stats['te'])}."
    else:
        sentence = (
            f"It has not been the yardstick. R-squared {pct(stats['r2'], 0)}, tracking error {pct(stats['te'])}."
        )
    if fund["plan"] == "regular":
        sentence += " It is the regular plan."
    elif fund["plan"] == "direct":
        sentence += " It is the direct plan."
    return sentence


def holdings_book(code: int) -> Optional[dict]:
    rows = HOLDINGS.get(code)
    if not rows:
        return None
    from datetime import date

    mapped = sum(weight for _, _, weight in rows)
    as_of = date.fromisoformat(HOLDINGS_AS_OF)
    days = (date.today() - as_of).days
    return {
        "rows": rows,
        "mapped": mapped,
        "unknown": max(0.0, 1 - mapped),
        "days": days,
        "stale": days > 31,
        "as_of": HOLDINGS_AS_OF,
    }


def herfindahl(weights: list[float]) -> Optional[float]:
    if not weights or any(weight < 0 for weight in weights):
        return None
    if abs(sum(weights) - 1) > 0.005:
        return None
    return sum(weight * weight for weight in weights)


def known_overlap(left: int, right: int) -> Optional[float]:
    a = {isin: weight for _, isin, weight in HOLDINGS.get(left, [])}
    b = {isin: weight for _, isin, weight in HOLDINGS.get(right, [])}
    if not a or not b:
        return None
    return sum(min(weight, b.get(isin, 0)) for isin, weight in a.items())


def holder_counts(code: int) -> list[tuple[str, int]]:
    rows = HOLDINGS.get(code) or []
    out = []
    for name, isin, _weight in rows:
        count = sum(1 for book in HOLDINGS.values() if any(row[1] == isin for row in book))
        out.append((name, count))
    return out


def historical_tail(returns: list[float]) -> Optional[float]:
    if len(returns) < 12:
        return None
    import math

    ordered = sorted(returns)
    index = max(0, math.ceil(0.05 * len(ordered)) - 1)
    return ordered[index]


def bootstrap_median(returns: list[float], seed: int = 17) -> Optional[float]:
    if len(returns) < 12:
        return None
    import random

    rng = random.Random(seed)
    endings = []
    for _ in range(400):
        value = 1.0
        for _month in range(12):
            value *= 1 + returns[rng.randrange(len(returns))]
        endings.append(value - 1)
    endings.sort()
    return endings[len(endings) // 2]


def _nav_on_or_before(rows: list, iso: str) -> Optional[float]:
    best = None
    best_key = ""
    for row in rows or []:
        parts = str(row.get("date", "")).split("-")
        if len(parts) != 3:
            continue
        key = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        if key > iso:
            continue
        try:
            nav = float(row["nav"])
        except (TypeError, ValueError):
            continue
        if nav > 0 and key > best_key:
            best = nav
            best_key = key
    return best


def xirr(flows: list[tuple[str, float]]) -> Optional[float]:
    if len(flows) < 2:
        return None
    from datetime import date

    dated = sorted((date.fromisoformat(iso), amount) for iso, amount in flows)
    t0 = dated[0][0]
    years = [(item[0] - t0).days / 365 for item in dated]

    def npv(rate: float) -> float:
        return sum(amount / (1 + rate) ** years[i] for i, (_day, amount) in enumerate(dated))

    def slope(rate: float) -> float:
        return sum(-years[i] * amount / (1 + rate) ** (years[i] + 1) for i, (_day, amount) in enumerate(dated))

    rate = 0.1
    for _ in range(50):
        d = slope(rate)
        if d == 0:
            return None
        nxt = rate - npv(rate) / d
        if nxt <= -0.99:
            return None
        if abs(nxt - rate) < 1e-8:
            return nxt
        rate = nxt
    return rate if abs(npv(rate)) < 1e-4 else None


def mix_held(lots: list[tuple[int, str, float, str]]) -> Optional[dict]:
    series = []
    for code, _name, _amount, _date in lots:
        payload = load_scheme(code)
        monthly = monthly_returns(month_end_navs(payload.get("data") or []))
        series.append(monthly[-36:])
    maps = [dict(rows) for rows in series]
    months = [ym for ym, _ret in series[0] if all(ym in item for item in maps)]
    if len(months) < 6:
        return None
    values = [amount for _code, _name, amount, _date in lots]
    returns = []
    for ym in months:
        start = sum(values)
        values = [value * (1 + maps[i][ym]) for i, value in enumerate(values)]
        end = sum(values)
        if start > 0:
            returns.append(end / start - 1)
    start_value = sum(amount for _c, _n, amount, _d in lots)
    end_value = sum(values)
    personal = None
    flows = []
    ending = 0.0
    end_iso = ""
    for code, _name, amount, bought in lots:
        payload = load_scheme(code)
        rows = payload.get("data") or []
        buy = _nav_on_or_before(rows, bought)
        latest = rows[-1] if rows else None
        if buy is None or not latest:
            flows = []
            break
        parts = str(latest["date"]).split("-")
        iso = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        ending += (amount / buy) * float(latest["nav"])
        end_iso = iso if iso > end_iso else end_iso
        flows.append((bought, -amount))
    if flows and ending > 0:
        flows.append((end_iso, ending))
        personal = xirr(flows)
    return {
        "start": months[0],
        "end": months[-1],
        "months": len(months),
        "start_value": start_value,
        "end_value": end_value,
        "annualised": annualised(end_value / start_value, len(months)),
        "vol": _stdev(returns) * (12 ** 0.5),
        "var": historical_tail(returns),
        "personal": personal,
    }


def run_ui() -> None:
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="the book", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
          .stApp { background: #f4f1ea; }
          [data-testid="stSidebar"] { background: #f4f1ea; }
          h1, h2, h3 { font-weight: 500; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for key, value in (("analysis", None), ("hits", []), ("error", ""), ("horizon", 12), ("amount", 100000)):
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.pop("go_book", False):
        st.session_state.room = ROOMS[0]
    if st.session_state.pop("clear_query", False):
        st.session_state.query = ""

    st.sidebar.markdown("## the book")
    room = st.sidebar.radio("Room", ROOMS, key="room", label_visibility="collapsed")

    analysis = st.session_state.analysis
    if (
        analysis
        and analysis["months"] != st.session_state.horizon
        and st.session_state.get("horizon_failed") != st.session_state.horizon
    ):
        with st.spinner("Reading the fund’s history…"):
            try:
                st.session_state.analysis = analyse(analysis["fund"]["code"], st.session_state.horizon)
                st.session_state.horizon_failed = None
                st.session_state.error = ""
            except Exception as exc:
                st.session_state.horizon_failed = st.session_state.horizon
                st.session_state.error = str(exc)
        analysis = st.session_state.analysis

    st.text_input(
        "Mutual fund",
        key="query",
        placeholder="Search a scheme, for example Parag Parikh",
    )
    if st.button("Search"):
        query = st.session_state.query.strip()
        if len(query) < 2:
            st.session_state.error = "Type at least two letters."
            st.session_state.hits = []
        else:
            try:
                st.session_state.hits = search_schemes(query)
                st.session_state.error = "" if st.session_state.hits else "No growth schemes matched."
            except Exception as exc:
                st.session_state.hits = []
                st.session_state.error = str(exc)

    for hit in list(st.session_state.hits):
        if st.button(hit["name"], key=f"hit-{hit['code']}"):
            with st.spinner("Reading the fund’s history…"):
                try:
                    st.session_state.analysis = analyse(hit["code"], st.session_state.horizon)
                    st.session_state.hits = []
                    st.session_state.error = ""
                    st.session_state.go_book = True
                    st.session_state.clear_query = True
                    st.rerun()
                except Exception as exc:
                    st.session_state.error = str(exc)

    if st.session_state.error:
        st.warning(st.session_state.error)

    analysis = st.session_state.analysis
    if analysis:
        st.radio(
            "Window",
            [12, 36, 60],
            format_func=lambda months: {12: "1 year", 36: "3 years", 60: "5 years"}[months],
            horizontal=True,
            key="horizon",
        )
        if analysis["paired"] < analysis["asked_months"]:
            if analysis["yardstick"]:
                st.caption(
                    f"This window only has {analysis['paired']} months in common with the yardstick, so it is shorter than the one you picked."
                )
            else:
                st.caption(f"This fund only has {analysis['paired']} months in the window you picked.")

    if not analysis and room != "Portfolio":
        st.title("Pick a fund. You do not need to upload anything.")
        st.write("The book is the scheme you select. A portfolio is optional, and research does not ask for a name.")
    elif room == "Overview" and analysis:
        st.caption(analysis["fund"]["category"])
        st.title(verdict(analysis))
        st.caption(f"{analysis['start']} to {analysis['end']}. {analysis['paired']} months. Source mfapi.in.")
        fund = analysis["fund"]
        st.write(f"**{fund['name']}**")
        st.write(f"{fund['house']} · {fund['plan']} plan · ISIN {fund['isin'] or 'not published'}")
    elif room == "Performance" and analysis:
        st.title("Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Annualised", pct(analysis["fund_return"]))
        c2.metric("Volatility", pct(analysis["vol"]))
        c3.metric("Growth of ₹1 lakh", inr(analysis["sample_growth"]))
        st.caption("The fund’s history over this window. Not your money, and not a forecast.")
        _path(st, analysis)
    elif room == "Benchmark" and analysis:
        st.title("The index it is measured against")
        st.write(analysis["yardstick_note"])
        if analysis["yardstick"]:
            st.caption(analysis["yardstick"]["name"])
        stats = analysis["index"]
        if not stats:
            st.caption("Not enough overlapping months with a yardstick.")
        else:
            left, right = st.columns(2)
            rows = [
                ("Beta", f"{stats['beta']:.2f}"),
                ("R-squared", pct(stats["r2"], 0)),
                ("Tracking error", pct(stats["te"])),
                ("Information ratio", f"{stats['ir']:.2f}"),
                ("Up capture", pct(stats["up_capture"], 0)),
                ("Down capture", pct(stats["down_capture"], 0)),
            ]
            for i, (label, value) in enumerate(rows):
                (left if i % 2 == 0 else right).metric(label, value)
    elif room == "Drivers" and analysis:
        st.title("Drivers")
        st.write("Method: Brinson-Fachler.")
        st.write(
            "Incomplete. Sector returns are not in the public feed, and the holdings file does not cover the whole fund. No effect is estimated."
        )
    elif room == "Risk" and analysis:
        st.title("When it hurts")
        dd = analysis["drawdown"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volatility", pct(analysis["vol"]))
        c2.metric("Worst fall", pct(None if dd is None else dd["max_dd"]))
        if dd is None:
            back = "—"
        elif dd["recovered"]:
            back = str(dd["recovery_months"])
        else:
            back = "Not yet"
        c3.metric("Months to get back", back)
        c4.metric("Monthly VaR", pct(historical_tail(analysis["returns"])))
        st.caption("Historical VaR, 95%, one month. Not a maximum loss. Sharpe uses a 6.5% stand-in, not a live Treasury bill.")
        _path(st, analysis)
    elif room == "Holdings" and analysis:
        book = holdings_book(analysis["fund"]["code"])
        st.title("Holdings")
        if not book:
            st.write("This scheme is not in the synthetic demo file. Weights are not invented for it.")
        else:
            st.warning(HOLDINGS_LABEL)
            stale = " Stale. Still shown, not hidden." if book["stale"] else ""
            st.caption(
                f"Demo date {book['as_of']}. {book['days']} days old.{stale} Mapped {pct(book['mapped'], 0)}. Unknown {pct(book['unknown'], 0)}."
            )
            st.table(
                [{"Security": name, "Weight": pct(weight, 0)} for name, _isin, weight in book["rows"]]
                + [{"Security": "Unknown", "Weight": pct(book["unknown"], 0)}]
            )
    elif room == "Crowding" and analysis:
        st.title("Crowding")
        book = holdings_book(analysis["fund"]["code"])
        if not book:
            st.write("This scheme is not in the synthetic file. A peer’s return is not a substitute.")
        else:
            other = 120716 if analysis["fund"]["code"] != 120716 else 122639
            overlap = known_overlap(analysis["fund"]["code"], other)
            st.caption(f"Synthetic file only. Sample of {len(HOLDINGS)} schemes.")
            st.write("HHI = sum of squared weights. Weights must sum to 1.")
            st.write(
                "Not calculated. Part of the book is unknown, and that slice is not treated as one holding or spread across the rest."
            )
            if overlap is not None:
                other_name = "UTI Nifty 50 Index Fund" if other == 120716 else "Parag Parikh Flexi Cap Fund"
                st.write(
                    f"Known-name overlap with {other_name}: {pct(overlap, 0)}. Unknown weight is left out, so this is not the full overlap."
                )
            st.table(
                [{"Security": name, "Schemes in the file": f"{count} of {len(HOLDINGS)}"} for name, count in holder_counts(analysis["fund"]["code"])]
            )
    elif room == "Peers" and analysis:
        st.title("Peers")
        st.caption(
            "Same category words, direct plan, growth option, this scheme left out. Not the official peer group, and not a rank."
        )
        _peer_table(st, analysis)
    elif room == "What-if" and analysis:
        st.title("What-if")
        st.write("No portfolio is required. The shock is this fund’s beta times a market move. It is not a forecast.")
        shock = st.number_input("Equity market move, percent", value=-10.0, step=1.0)
        beta = None if analysis["index"] is None else analysis["index"]["beta"]
        if beta is None:
            st.header("—")
            st.caption("No beta in this window, so the estimate is off.")
        else:
            st.header(pct(beta * shock / 100))
            st.caption(f"Beta {beta:.2f} times {shock:.1f}%.")
        st.caption("Model. Each path resamples this fund’s own monthly returns. Seed 17. Not a forecast.")
        middle = bootstrap_median(analysis["returns"])
        st.metric("Middle path, 12 months", pct(middle))
    elif room == "Portfolio":
        st.title("A portfolio, if you want one.")
        st.write("Research still works with no portfolio and no name. This mix is not one fund’s CAGR.")
        if st.button("Load a synthetic book"):
            st.session_state.lots = list(DUMMY_LOTS)
        lots = st.session_state.get("lots") or []
        if lots:
            st.caption("Synthetic demo. Round amounts and invented dates. Not a person’s account.")
            st.table(
                [{"Fund": name, "Amount": inr(amount), "Date": bought} for _code, name, amount, bought in lots]
            )
            if st.button("Read this mix"):
                with st.spinner("Reading the funds…"):
                    try:
                        st.session_state.mix = mix_held(lots)
                        st.session_state.mix_error = ""
                    except Exception as exc:
                        st.session_state.mix = None
                        st.session_state.mix_error = str(exc)
        mix = st.session_state.get("mix")
        if st.session_state.get("mix_error"):
            st.warning(st.session_state.mix_error)
        if mix:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Put in", inr(mix["start_value"]))
            c2.metric("Held to the end", inr(mix["end_value"]))
            c3.metric("Time-weighted", pct(mix["annualised"]))
            c4.metric("Money-weighted", pct(mix["personal"]))
            st.caption(
                f"{mix['start']} to {mix['end']}, {mix['months']} shared months. Time-weighted is the mix. Money-weighted uses the dates. Neither is one fund’s CAGR."
            )
            st.caption(f"Mix volatility {pct(mix['vol'])}. Monthly VaR {pct(mix['var'])}. Historical, 95%, one month. Not a maximum loss.")
    else:
        st.title("Pick a fund first.")

    st.divider()
    st.caption(
        "These figures describe the window you picked. They are the fund’s history, not your money, and not a forecast. The yardstick is an index fund, not the official total-return index."
    )


def _path(st, analysis: dict) -> None:
    import plotly.graph_objects as go

    series = analysis["series"]
    if len(series) < 2:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[row["ym"] for row in series],
            y=[row["fund"] for row in series],
            name="Fund",
            line={"color": "#1a1814", "width": 2},
        )
    )
    if analysis["yardstick"] and analysis["index"]:
        fig.add_trace(
            go.Scatter(
                x=[row["ym"] for row in series],
                y=[row["bench"] for row in series],
                name="Yardstick",
                line={"color": "#1f6f66", "width": 1.5},
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="#fbf9f4",
        plot_bgcolor="#fbf9f4",
        legend={"orientation": "h"},
        yaxis_title="Growth of one rupee",
    )
    st.plotly_chart(fig, use_container_width=True)


def _peer_table(st, analysis: dict) -> None:
    with st.spinner("Reading a few other funds…"):
        try:
            rows = peers_for(analysis)
        except Exception:
            rows = []
    table = [
        {
            "Fund": "This fund",
            "Return": pct(analysis["fund_return"]),
            "Volatility": pct(analysis["vol"]),
            "Worst fall": pct(None if analysis["drawdown"] is None else analysis["drawdown"]["max_dd"]),
        }
    ]
    for row in rows:
        table.append(
            {
                "Fund": row["name"],
                "Return": pct(row["ann_return"]),
                "Volatility": pct(row["vol"]),
                "Worst fall": pct(row["max_dd"]),
            }
        )
    if len(table) == 1:
        st.caption("Sample of 0. No peer funds came back from the feed.")
    else:
        st.caption(f"Sample of {len(table) - 1}.")
    st.table(table)


if os.environ.get("BOOK_SELFTEST") == "1":
    book = holdings_book(122639)
    assert book is not None and book["unknown"] > 0.5
    assert herfindahl([row[2] for row in book["rows"]]) is None
    assert abs((known_overlap(122639, 120716) or 0) - 0.30) < 1e-9
    assert abs((herfindahl([0.5, 0.5]) or 0) - 0.5) < 1e-12
    sample = analyse(122639, 12)
    print(sample["fund"]["name"])
    print(sample["fund"]["category"], sample["equity"], sample["paired"], sample["start"], sample["end"])
    print(sample["yardstick"])
    print(sample["index"])
    print(verdict(sample))
    assert sample["index"] is not None
    assert 0 < sample["index"]["beta"] < 2
    assert 0 < sample["index"]["r2"] <= 1
    assert sample["paired"] >= 6
    assert "Nifty 500" in sample["yardstick"]["name"]
else:
    run_ui()
