"""Demo app: python -m streamlit run app/Home.py"""

import streamlit as st

import style

st.set_page_config(page_title="Foundation models under hospital shift", layout="centered")
style.apply()

pages = [
    st.Page("views/patches.py", title="Patches", default=True),
    st.Page("views/results.py", title="Results"),
    st.Page("views/about.py", title="Method"),
]
st.navigation(pages, position="top").run()
