import streamlit.components.v1 as components
from ui import configure_page, page_header, sidebar

configure_page("Dataset Health")
sidebar(active=None)
page_header("Dataset Health", "Composition, provenance, and pass-rate trends.")
components.iframe("http://localhost:8502", height=1200, scrolling=True)