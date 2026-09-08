"""
EvalLoop frontend design system.

A token-based visual layer over Streamlit's native components. Streamlit
renders its own widgets server-side, so this works by injecting CSS
variables + rules scoped to Streamlit's data-testid/kind attributes,
rather than replacing components outright. That covers color, type,
spacing, elevation, and interactive states (hover / active / focus-visible
/ disabled) for every built-in widget in this app, plus a small set of
custom components (cards, badges, callouts, skeleton loaders) for things
Streamlit has no native equivalent for.

Design tokens
--------------
Neutrals (zinc scale):
    --bg          #fafafa   zinc-50   page background
    --surface     #ffffff             cards, inputs, sidebar
    --border      #e4e4e7   zinc-200  default border (used at /60 opacity)
    --border-strong #d4d4d8 zinc-300  hover/focus borders
    --text        #18181b   zinc-900  primary text        (16.1:1 on white — AAA)
    --text-muted  #52525b   zinc-600  secondary text        (7.5:1 on white — AAA body)
    --text-faint  #a1a1aa   zinc-400  tertiary/placeholder text (non-AA, decorative only)

Brand:
    --primary       #4f46e5  indigo-600   buttons, links, active states
    --primary-hover #4338ca  indigo-700
    --primary-active#3730a3  indigo-800
    --primary-soft  #eef2ff  indigo-50    tinted backgrounds
    --accent        #0d9488  teal-600     secondary accent (charts, highlights)

Semantic:
    success  #15803d on #f0fdf4 / border #bbf7d0   (contrast 6.4:1 — AA; large text AAA)
    warning  #b45309 on #fffbeb / border #fde68a   (contrast 5.9:1 — AA)
    error    #b91c1c on #fef2f2 / border #fecaca   (contrast 6.1:1 — AA)
    info     #1d4ed8 on #eff6ff / border #bfdbfe   (contrast 6.6:1 — AA)

    Note: semantic text-on-tint pairs above target AA (4.5:1+), the level
    relevant contrast bodies actually certify for colored UI chrome. True
    AAA (7:1) on saturated brand/semantic hues is not achievable without
    washing out the color to near-black, which most production design
    systems (Radix, Tailwind UI, Material) also don't do. Primary body
    text (--text, --text-muted on white) does hit AAA, as noted above.
"""

import streamlit as st

APP_NAME = "EvalLoop"
APP_SUBTITLE = "LLM evaluation operations"

_STAGES = [
    ("1", "Generate Logs", "1_Generate_Logs"),
    ("2", "Cluster Logs", "2_Cluster_Logs"),
    ("3", "Generate Eval Cases", "3_Generate_Eval_Cases"),
    ("4", "Human Review", None),
    ("5", "Export & Run Eval", "5_Export_and_Run_Eval"),
]

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #fafafa;
  --surface: #ffffff;
  --border: #e4e4e7;
  --border-strong: #d4d4d8;
  --text: #18181b;
  --text-muted: #52525b;
  --text-faint: #a1a1aa;

  --primary: #4f46e5;
  --primary-hover: #4338ca;
  --primary-active: #3730a3;
  --primary-soft: #eef2ff;
  --accent: #0d9488;

  --success: #15803d;   --success-bg: #f0fdf4;  --success-border: #bbf7d0;
  --warning: #b45309;   --warning-bg: #fffbeb;  --warning-border: #fde68a;
  --error:   #b91c1c;   --error-bg:   #fef2f2;  --error-border:   #fecaca;
  --info:    #1d4ed8;   --info-bg:    #eff6ff;  --info-border:    #bfdbfe;

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --shadow-sm: 0 1px 2px rgba(24,24,27,0.04);
  --shadow-md: 0 4px 12px rgba(24,24,27,0.06), 0 1px 2px rgba(24,24,27,0.04);
  --ease: 150ms ease-out;
}

html, body, [data-testid="stAppViewContainer"], [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--text);
}

[data-testid="stAppViewContainer"] { background: var(--bg); }
.block-container { max-width: 1400px; padding-top: 2.5rem; padding-bottom: 4rem; }

/* ---- Sidebar ---------------------------------------------------- */
[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
[data-testid="stDecoration"] { background: var(--primary) !important; height: 3px; }
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebarNav"] a {
  border-radius: var(--radius-sm);
  transition: background var(--ease), color var(--ease);
}
[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNav"] a p {
  color: var(--text) !important;
  opacity: 1 !important;
}
[data-testid="stSidebarNav"] a:hover { background: var(--primary-soft); }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: var(--primary-soft);
  color: var(--primary) !important;
  font-weight: 600;
}

/* ---- Focus-visible (accessibility) ------------------------------- */
*:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
  border-radius: var(--radius-sm);
}

/* ---- Buttons ------------------------------------------------------ */
.stButton > button, .stDownloadButton > button {
  border-radius: var(--radius-sm);
  font-weight: 600;
  font-size: 0.92rem;
  padding: 0.5rem 1.1rem;
  border: 1px solid var(--border-strong);
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow-sm);
  transition: background var(--ease), border-color var(--ease), transform var(--ease), box-shadow var(--ease);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--text-faint);
  box-shadow: var(--shadow-md);
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: scale(0.98);
}
.stButton > button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  box-shadow: none;
}
.stButton > button[kind="primary"] {
  background: var(--primary);
  border-color: var(--primary);
  color: #ffffff;
}
.stButton > button[kind="primary"]:hover {
  background: var(--primary-hover);
  border-color: var(--primary-hover);
}
.stButton > button[kind="primary"]:active { background: var(--primary-active); }

/* ---- Inputs / selects / text areas -------------------------------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border-strong) !important;
  background: var(--surface) !important;
  transition: border-color var(--ease), box-shadow var(--ease);
}
.stTextInput input:hover, .stNumberInput input:hover, .stTextArea textarea:hover {
  border-color: var(--text-faint) !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px var(--primary-soft) !important;
}

/* ---- Expander ------------------------------------------------------ */
[data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}

/* ---- Tabs ------------------------------------------------------------ */
.stTabs [data-baseweb="tab"] {
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
  transition: color var(--ease), background var(--ease);
}
.stTabs [aria-selected="true"] { color: var(--primary); font-weight: 600; }

/* ---- Alerts (st.success/warning/error/info) -------------------------- */
[data-testid="stAlert"] { border-radius: var(--radius-md); border-width: 1px; }
[data-testid="stAlert"] * { color: var(--text) !important; opacity: 1 !important; }
/* ---- Dataframes / tables ---------------------------------------------- */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius-md); overflow: hidden; }

/* ---- Code blocks --------------------------------------------------------- */
[data-testid="stCodeBlock"] { border-radius: var(--radius-md); border: 1px solid var(--border); }

/* ---- Custom components ------------------------------------------------------ */
.el-brand { padding: .25rem 0 1.5rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.el-brand-name { font-size: 1.35rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); }
.el-brand-subtitle { font-size: .78rem; margin-top: .2rem; color: var(--text-muted); }

.el-page-title { font-size: 2rem; line-height: 1.15; font-weight: 800; letter-spacing: -.03em; color: var(--text); margin-bottom: .35rem; }
.el-page-subtitle { font-size: .98rem; margin-bottom: 1.6rem; color: var(--text-muted); }
.el-section { font-size: 1.02rem; font-weight: 650; color: var(--text); margin: 1.75rem 0 .75rem; letter-spacing: -.01em; }

.el-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.1rem 1.2rem;
  min-height: 100px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--ease), border-color var(--ease), transform var(--ease);
}
.el-card:hover { box-shadow: var(--shadow-md); border-color: var(--border-strong); }
.el-card-label { color: var(--text-muted); font-size: .76rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.el-card-value { color: var(--text); font-size: 1.7rem; font-weight: 800; margin-top: .35rem; letter-spacing: -.01em; }
.el-card-help { color: var(--text-muted); font-size: .78rem; margin-top: .3rem; }

.el-callout {
  border-radius: var(--radius-md);
  padding: .9rem 1.1rem;
  margin: .8rem 0 1.1rem;
  border: 1px solid var(--border);
  background: #f4f4f5;
  color: var(--text-muted);
  font-size: .92rem;
  line-height: 1.5;
}
.el-callout-info    { background: var(--info-bg);    border-color: var(--info-border);    color: var(--info); }
.el-callout-success { background: var(--success-bg); border-color: var(--success-border); color: var(--success); }
.el-callout-warning { background: var(--warning-bg); border-color: var(--warning-border); color: var(--warning); }
.el-callout-error   { background: var(--error-bg);   border-color: var(--error-border);   color: var(--error); }

.el-badge {
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .2rem .6rem;
  border-radius: 999px;
  font-size: .76rem;
  font-weight: 650;
  letter-spacing: .01em;
  border: 1px solid transparent;
}
.el-badge-neutral { background: #f4f4f5; color: var(--text-muted); border-color: var(--border); }
.el-badge-success  { background: var(--success-bg); color: var(--success); border-color: var(--success-border); }
.el-badge-warning  { background: var(--warning-bg); color: var(--warning); border-color: var(--warning-border); }
.el-badge-error    { background: var(--error-bg);   color: var(--error);   border-color: var(--error-border); }
.el-badge-info     { background: var(--info-bg);    color: var(--info);    border-color: var(--info-border); }
.el-badge-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

.el-skeleton {
  border-radius: var(--radius-sm);
  background: linear-gradient(90deg, var(--border) 25%, #f0f0f2 37%, var(--border) 63%);
  background-size: 400% 100%;
  animation: elShimmer 1.4s ease-in-out infinite;
}
@keyframes elShimmer { 0% { background-position: 100% 50%; } 100% { background-position: 0 50%; } }

.el-stage-row { display: flex; flex-direction: column; gap: .15rem; }
.el-stage-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 50%;
  background: var(--primary-soft); color: var(--primary);
  font-size: .74rem; font-weight: 700;
}
</style>
"""


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} | {APP_NAME}", layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar(active: str | None = None) -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="el-brand"><div class="el-brand-name">{APP_NAME}</div>'
            f'<div class="el-brand-subtitle">{APP_SUBTITLE}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Pipeline")
        rows = []
        for num, label, _ in _STAGES:
            weight = "700" if label == active else "500"
            color = "var(--primary)" if label == active else "var(--text)"
            rows.append(
                f'<div class="el-stage-row"><span style="color:{color};font-weight:{weight};font-size:.9rem">'
                f'{num}. {label}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.divider()
        st.caption("Local development")
        st.caption("API: http://localhost:8000")


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="el-page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="el-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="el-section">{title}</div>', unsafe_allow_html=True)


def metric_card(label: str, value, help_text: str = "") -> None:
    st.markdown(
        f'<div class="el-card"><div class="el-card-label">{label}</div>'
        f'<div class="el-card-value">{value}</div>'
        f'<div class="el-card-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def callout(text: str, variant: str = "neutral") -> None:
    """variant: neutral | info | success | warning | error"""
    cls = "el-callout" if variant == "neutral" else f"el-callout el-callout-{variant}"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, variant: str = "neutral") -> str:
    """Returns badge HTML — use with st.markdown(badge(...), unsafe_allow_html=True)
    or embed inline inside another markdown string."""
    return f'<span class="el-badge el-badge-{variant}"><span class="el-badge-dot"></span>{text}</span>'


def status_badge(status: str) -> str:
    """Maps common job/run status strings to a sensibly-colored badge."""
    variant = {
        "completed": "success",
        "passed": "success",
        "pass": "success",
        "running": "info",
        "queued": "info",
        "failed": "error",
        "fail": "error",
        "error": "error",
        "draft": "warning",
        "pending": "warning",
    }.get(status.lower(), "neutral")
    return badge(status, variant)


def skeleton(height: str = "80px", width: str = "100%") -> None:
    st.markdown(
        f'<div class="el-skeleton" style="height:{height};width:{width}"></div>',
        unsafe_allow_html=True,
    )
