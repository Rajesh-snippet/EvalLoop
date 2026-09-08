import streamlit.components.v1 as components
from ui import configure_page, page_header, sidebar

configure_page("Human Review")
sidebar(active="Human Review")
page_header("Human Review", "Phase 4 — embedded review workspace (runs on its own Streamlit server).")
components.iframe("http://localhost:8501", height=900, scrolling=True)