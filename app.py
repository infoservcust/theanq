import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import time

st.set_page_config(page_title="MF Look-through & Overlap", layout="wide")

st.title("Indian Mutual Fund Look-through & Overlap Analyzer")
st.caption("Primary-level tool using public data. Shows true exposure and concentration risk across your funds.")

# ---------- Helpers ----------
@st.cache_data(ttl=3600)
def search_schemes(query: str) -> List[Dict]:
    if not query or len(query) < 3:
        return []
    try:
        r = requests.get(f"https://api.mfapi.in/mf/search?q={query}", timeout=10)
        r.raise_for_status()
        return r.json()[:15]  # limit results
    except Exception as e:
        st.warning(f"Scheme search temporarily unavailable: {e}")
        return []

def demo_holdings() -> Dict[str, pd.DataFrame]:
    """Sample holdings for demo mode so visualizations work immediately."""
    data = {
        "Parag Parikh Flexi Cap": pd.DataFrame({
            "name": ["HDFC Bank", "ICICI Bank", "Reliance Industries", "Infosys", "TCS", "ITC", "Bajaj Finance"],
            "weight": [8.5, 7.2, 6.1, 5.4, 4.8, 4.2, 3.9]
        }),
        "HDFC Flexi Cap": pd.DataFrame({
            "name": ["HDFC Bank", "Reliance Industries", "ICICI Bank", "Infosys", "Larsen & Toubro", "SBI", "Axis Bank"],
            "weight": [9.1, 7.8, 6.5, 5.2, 4.7, 4.1, 3.6]
        }),
        "SBI Bluechip": pd.DataFrame({
            "name": ["HDFC Bank", "Reliance Industries", "ICICI Bank", "Infosys", "TCS", "Hindustan Unilever", "Bajaj Finance"],
            "weight": [10.2, 8.4, 7.1, 6.0, 5.3, 4.5, 3.8]
        }),
    }
    return data

def compute_lookthrough(selected: List[Dict], allocations: Dict[str, float], holdings_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    total = sum(allocations.values()) or 1
    rows = []
    for fund in selected:
        name = fund["schemeName"]
        alloc_pct = allocations.get(name, 0) / total
        if name not in holdings_map:
            continue
        df = holdings_map[name]
        for _, row in df.iterrows():
            rows.append({
                "stock": row["name"],
                "fund": name,
                "weight_in_fund": row["weight"],
                "contribution": row["weight"] * alloc_pct
            })
    if not rows:
        return pd.DataFrame()
    lt = pd.DataFrame(rows)
    agg = lt.groupby("stock")["contribution"].sum().reset_index()
    agg = agg.sort_values("contribution", ascending=False)
    agg["portfolio_weight"] = (agg["contribution"] / agg["contribution"].sum() * 100).round(2)
    return agg

def compute_overlap(holdings_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    funds = list(holdings_map.keys())
    if len(funds) < 2:
        return pd.DataFrame()
    matrix = pd.DataFrame(index=funds, columns=funds, dtype=float)
    for f1 in funds:
        for f2 in funds:
            if f1 == f2:
                matrix.loc[f1, f2] = 100.0
                continue
            h1 = holdings_map[f1].set_index("name")["weight"]
            h2 = holdings_map[f2].set_index("name")["weight"]
            common = set(h1.index) & set(h2.index)
            overlap = sum(min(h1.get(s, 0), h2.get(s, 0)) for s in common)
            matrix.loc[f1, f2] = round(overlap, 1)
    return matrix

# ---------- UI ----------
mode = st.radio("Mode", ["Demo (sample data)", "Live search (requires holdings source)"], horizontal=True)

if mode.startswith("Demo"):
    st.info("Demo mode uses sample holdings so you can see the visualizations immediately. Switch to Live when a holdings API is connected.")
    holdings_map = demo_holdings()
    selected = [{"schemeName": k, "schemeCode": 0} for k in holdings_map.keys()]
else:
    query = st.text_input("Search mutual fund schemes", placeholder="e.g. Parag Parikh, HDFC Flexi, SBI Bluechip")
    results = search_schemes(query)
    if results:
        options = {r["schemeName"]: r for r in results}
        chosen = st.multiselect("Select schemes (up to 8)", options=list(options.keys()), max_selections=8)
        selected = [options[c] for c in chosen]
    else:
        selected = []
    holdings_map = {}  # In live mode we would fetch here; placeholder for now
    if selected:
        st.warning("Live holdings fetch is not yet connected (community APIs are currently unstable). Use Demo mode to explore the UI, or we can add Excel upload next.")

if selected:
    st.subheader("Your allocations")
    allocations = {}
    cols = st.columns(min(len(selected), 4))
    for i, fund in enumerate(selected):
        name = fund["schemeName"]
        with cols[i % len(cols)]:
            amt = st.number_input(f"{name[:40]}...", min_value=0.0, value=100000.0, step=10000.0, key=name)
            allocations[name] = amt

    if st.button("Analyze Portfolio", type="primary") and sum(allocations.values()) > 0:
        with st.spinner("Calculating..."):
            time.sleep(0.5)
            lt = compute_lookthrough(selected, allocations, holdings_map)
            overlap = compute_overlap(holdings_map)

        if not lt.empty:
            st.subheader("True Look-through Portfolio (Top holdings)")
            st.dataframe(lt.head(20), use_container_width=True)
            fig = px.bar(lt.head(15), x="portfolio_weight", y="stock", orientation="h",
                         title="Top stocks by true portfolio weight")
            st.plotly_chart(fig, use_container_width=True)

        if not overlap.empty:
            st.subheader("Pairwise Weighted Overlap (%)")
            st.caption("Higher number = more common holdings. 30%+ means significant concentration risk.")
            fig2 = px.imshow(overlap.astype(float), text_auto=True, aspect="auto",
                             color_continuous_scale="Reds", title="Overlap Heatmap")
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(overlap, use_container_width=True)

        # Top overlapping securities across the portfolio
        if len(holdings_map) >= 2:
            all_stocks = set()
            for df in holdings_map.values():
                all_stocks.update(df["name"].tolist())
            counts = []
            for stock in all_stocks:
                funds_holding = [f for f, df in holdings_map.items() if stock in df["name"].values]
                if len(funds_holding) > 1:
                    avg_w = sum(df[df["name"]==stock]["weight"].iloc[0] for f, df in holdings_map.items() if stock in df["name"].values) / len(funds_holding)
                    counts.append({"stock": stock, "num_funds": len(funds_holding), "avg_weight": round(avg_w, 1)})
            if counts:
                top_overlap = pd.DataFrame(counts).sort_values(["num_funds", "avg_weight"], ascending=False)
                st.subheader("Stocks held by multiple of your funds")
                st.dataframe(top_overlap.head(15), use_container_width=True)

else:
    st.info("Search and select schemes, or switch to Demo mode to see sample results.")

st.markdown("---")
st.caption("Data sources: MFapi.in for scheme search. Holdings in live mode will use public AMC disclosures when a stable free API is available. This is a primary-level tool for 1+ year horizons. Not investment advice.")
