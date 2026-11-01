# Indian Mutual Fund Look-through & Overlap Analyzer

Simple Streamlit web app for retail investors to see true look-through exposure and overlap across their mutual funds.

## Features (v1)
- Search Indian mutual fund schemes (via free MFapi.in)
- Enter allocations
- Look-through aggregated portfolio weights
- Pairwise weighted overlap heatmap
- Top stocks appearing in multiple funds
- Demo mode with sample data so visualizations work immediately

## Deploy on Streamlit Community Cloud (Free)

1. Create a free GitHub account if you don't have one.
2. Create a new public repository (e.g. `mf-overlap`).
3. Upload these files to the repository:
   - `app.py`
   - `requirements.txt`
4. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
5. Click "New app", select your repository, branch `main`, and main file `app.py`.
6. Deploy. You will get a free public URL (e.g. `https://yourname-mf-overlap.streamlit.app`).
7. Later you can point your own domain to it via Streamlit settings or a reverse proxy.

## Local run (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Demo mode works fully offline for visualizations.
- Live holdings currently show a warning because free community holdings APIs are unstable. The calculation engine is ready; we can add Excel upload or a stable API next.
- Built for Indian market, 1+ year horizon, zero coding required after deployment.

Not investment advice.
