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
    "Book",
    "What you own",
    "Cost",
    "The index",
    "Why it moved",
    "Risk",
    "Crowding",
    "Decide",
    "A fund",
    "How to read this",
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
        if len(rows) == 3:
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

    if not analysis:
        st.title("Pick a fund. You do not need to upload anything.")
        st.write(
            "The book is the scheme you select. What you own, and the rupee cost of money you have not put in, stay empty until you have a holding."
        )
    elif room == "Book":
        st.caption(analysis["fund"]["category"])
        st.title(verdict(analysis))
        st.caption(f"{analysis['start']} to {analysis['end']}. {analysis['paired']} months.")
    elif room == "What you own":
        st.title("What you own")
        st.write(
            "You have not entered a holding. This room is the stocks behind money you already have, and the rupees counted twice across your funds. The scheme you picked is not that."
        )
    elif room == "Cost":
        plan = analysis["fund"]["plan"]
        st.title("What it charges")
        shown = "plan we could not classify" if plan == "unknown" else f"{plan} plan"
        st.write(
            f"This is the {shown}. The public NAV feed does not publish the expense ratio, so rupees a year are not invented."
        )
        sibling = analysis["sibling"]
        if sibling and st.button(f"Open the {'regular' if plan == 'direct' else 'direct'} plan"):
            with st.spinner("Reading the fund’s history…"):
                try:
                    st.session_state.analysis = analyse(sibling["code"], st.session_state.horizon)
                    st.session_state.go_book = True
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        elif not sibling:
            st.caption("No matching other plan showed up in the search.")
    elif room == "The index":
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
                ("Active return", pct(stats["active"])),
                ("Information ratio", f"{stats['ir']:.2f}"),
                ("Up capture", pct(stats["up_capture"], 0)),
                ("Down capture", pct(stats["down_capture"], 0)),
                (
                    "Beta when the yardstick fell",
                    f"{stats['down_beta']:.2f}"
                    if stats["down_beta"] is not None
                    else f"Needs 4 down months. This window has {stats['down_months']}.",
                ),
            ]
            for i, (label, value) in enumerate(rows):
                (left if i % 2 == 0 else right).metric(label, value)
    elif room == "Why it moved":
        st.title("Why it moved")
        st.write(
            "The split into market, size, value and momentum needs three years and the IIMA factor file. That file is not in this release, so the leftover is not shown. A peer table is not that split."
        )
        st.caption(
            "Beside it, a few other direct growth funds the search returned for the same category. Not the full peer group, and not a rank."
        )
        _peer_table(st, analysis)
    elif room == "Risk":
        st.title("When it hurts")
        dd = analysis["drawdown"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Volatility", pct(analysis["vol"]))
        c2.metric("Worst fall", pct(None if dd is None else dd["max_dd"]))
        if dd is None:
            back = "—"
        elif dd["recovered"]:
            back = str(dd["recovery_months"])
        else:
            back = "Not yet"
        c3.metric("Months to get back", back)
        series = analysis["series"]
        if len(series) > 1:
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
        _peer_table(st, analysis)
    elif room == "Crowding":
        st.title("Crowding")
        st.write(
            "Crowding is how many schemes hold the same stocks. That needs the monthly holdings file, which is not in this release. A peer’s return is not a substitute, so nothing is shown in its place."
        )
    elif room == "Decide":
        st.title("If you had put money in")
        amount = st.number_input("Amount, in rupees", min_value=1000, step=10000, key="amount")
        grown = (analysis["sample_growth"] / 100000) * float(amount)
        st.header(inr(grown))
        st.write(
            f"{inr(float(amount))} at the start of this window would have been worth this at the end. That is the past. It is not a forecast, and it is not a recommendation."
        )
        st.caption("The fee in rupees needs the expense ratio. This feed does not publish it.")
    elif room == "A fund":
        st.title("This fund")
        fund = analysis["fund"]
        st.write(f"**Name.** {fund['name']}")
        st.write(f"**House.** {fund['house']}")
        st.write(f"**Category.** {fund['category']}")
        st.write(f"**ISIN.** {fund['isin'] or 'Not published'}")
        st.write(f"**Plan.** {fund['plan']}")
    else:
        st.title("How to read this")
        cards = [
            ("Beta", "How many points the fund moved when the yardstick moved one. It describes the window. It is not next year’s beta."),
            ("Tracking error", "How wide the monthly gap versus the yardstick was, as a yearly figure. Over 3 points, it was not hugging the yardstick."),
            ("R-squared", "How much of the fund’s movement the yardstick explains. Under 0.90, calling it “the index” is a stretch."),
            ("The yardstick", "An index fund, not the official total-return index. The gap includes that fund’s fee. A large-cap fund is set against a Nifty 50 fund because this feed has no Nifty 100 series."),
            ("Peers", "A few direct-growth funds the search returned. Not a rank, not a star, and not the thing the sentence is judged against."),
            ("The amount", "What a sum would have become over the window you picked. The past. Not a forecast, and not your SIP."),
        ]
        for title, body in cards:
            st.subheader(title)
            st.write(body)

    st.divider()
    st.caption(
        "These figures describe the window you picked. They are not a forecast. If you have not told us what you hold, this is the fund, not your money."
    )


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
        st.caption("No peer funds came back from the feed.")
    st.table(table)


if os.environ.get("BOOK_SELFTEST") == "1":
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
