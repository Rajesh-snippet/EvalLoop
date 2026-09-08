import streamlit as st
APP_NAME = "EvalLoop"
APP_SUBTITLE = "LLM evaluation operations"

def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} | {APP_NAME}", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]{background:#f7f8fa}
    [data-testid="stSidebar"]{border-right:1px solid #d9dee7}
    .block-container{max-width:1400px;padding-top:2rem;padding-bottom:3rem}
    .el-brand{padding:.25rem 0 1.5rem;border-bottom:1px solid #d9dee7;margin-bottom:1.5rem}
    .el-brand-name{font-size:1.35rem;font-weight:700;letter-spacing:-.02em;color:#101828}
    .el-brand-subtitle,.el-page-subtitle{color:#667085}
    .el-brand-subtitle{font-size:.78rem;margin-top:.2rem}
    .el-page-title{font-size:2rem;line-height:1.15;font-weight:700;letter-spacing:-.03em;color:#101828;margin-bottom:.35rem}
    .el-page-subtitle{font-size:.98rem;margin-bottom:1.6rem}
    .el-section{font-size:1.05rem;font-weight:650;color:#101828;margin:1.5rem 0 .7rem}
    .el-card{background:#fff;border:1px solid #d9dee7;border-radius:10px;padding:1rem 1.1rem;min-height:100px}
    .el-card-label{color:#667085;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
    .el-card-value{color:#101828;font-size:1.65rem;font-weight:700;margin-top:.35rem}
    .el-card-help{color:#667085;font-size:.78rem;margin-top:.25rem}
    .el-callout{background:#f2f4f7;border:1px solid #d9dee7;border-radius:10px;padding:.9rem 1rem;color:#344054;margin:.8rem 0 1rem}
    .stButton>button,.stDownloadButton>button{border-radius:7px;font-weight:600}
    </style>
    """, unsafe_allow_html=True)

def sidebar() -> None:
    with st.sidebar:
        st.markdown(f'<div class="el-brand"><div class="el-brand-name">{APP_NAME}</div><div class="el-brand-subtitle">{APP_SUBTITLE}</div></div>', unsafe_allow_html=True)
        st.caption("Pipeline")
        st.markdown("**1. Generate Logs**  \n**2. Cluster Logs**  \n**3. Generate Eval Cases**  \n**4. Human Review**  \n**5. Export & Run Eval**")
        st.divider()
        st.caption("Local development")
        st.caption("API: http://localhost:8000")

def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="el-page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="el-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def section(title: str) -> None:
    st.markdown(f'<div class="el-section">{title}</div>', unsafe_allow_html=True)

def metric_card(label: str, value, help_text: str = "") -> None:
    st.markdown(f'<div class="el-card"><div class="el-card-label">{label}</div><div class="el-card-value">{value}</div><div class="el-card-help">{help_text}</div></div>', unsafe_allow_html=True)
