"""Demo app: python -m streamlit run app/Home.py"""

import streamlit as st

st.set_page_config(page_title="foundation-shift", layout="wide")
st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  .patch-caption { font-size: 0.85rem; line-height: 1.35; margin-top: 0.25rem; }
  .right { color: #1baf7a; }
  .wrong { color: #e34948; }
  .small { font-size: 0.9rem; color: #8a8985; }
</style>
""", unsafe_allow_html=True)

pages = [
    st.Page("views/patches.py", title="Browse patches", default=True),
    st.Page("views/results.py", title="Results"),
    st.Page("views/about.py", title="About"),
]
st.navigation(pages).run()
