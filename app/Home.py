"""Portfolio Risk Lab entry point (portfolio builder lands here in M5)."""

import streamlit as st

import _bootstrap  # noqa: F401  (adds ../src to sys.path)
from components.layout import disclaimer_footer, page_setup
from risklab import __version__

page_setup()

st.title("Portfolio Risk Lab")
st.markdown(
    "How can this portfolio fail? Classical risk analytics, honestly evaluated ML on "
    "volatility and regimes, and cited answers from SEC 10-K risk factors."
)
st.info(
    "Build in progress: the portfolio builder and analysis pages arrive in later milestones.",
    icon=":material/construction:",
)
st.caption(f"Version {__version__}")

disclaimer_footer()
