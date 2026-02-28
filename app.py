"""
Streamlit frontend for the AI-Powered Brain MRI Assistant.
Light clinical dashboard — matches the blue/white medical mockup strictly.
"""

import base64
import os
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Brain MRI Assistant",
    page_icon="🧠",
    layout="wide",
)

ENABLE_VISUALIZATION = False

# ── Theme colours (hex only — no CSS vars in widget selectors) ──────────────
NAV   = "#1b3a6b"   # dark navy
MID   = "#254e96"   # mid-navy
BLUE  = "#d6e4f7"   # light blue tint
PALE  = "#eef4fc"   # very pale blue
ACC   = "#3b82f6"   # bright blue
TXT   = "#1a2744"   # near-black
TXTS  = "#3d526e"   # secondary text
TXTT  = "#6b7a94"   # tertiary / labels
BDR   = "#dce8f5"   # border
BGPG  = "#e8f0fb"   # page bg
BGCD  = "#ffffff"   # card bg
BGIN  = "#f0f5fc"   # inner bg

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* ── Global font ── */
    html, body, [class*="css"], .stApp, .stApp * {{
        font-family: 'DM Sans', sans-serif !important;
    }}

    /* ── Page background ── */
    .stApp {{ background: {BGPG} !important; }}
    .block-container {{
        padding: 1.6rem 1.8rem 2rem !important;
        max-width: 100% !important;
    }}

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header,
    [data-testid="stDecoration"],
    [data-testid="stToolbar"] {{ display: none !important; }}

    /* ══════════════════════════════════════════
       ALL NATIVE STREAMLIT TEXT — explicit hex
    ══════════════════════════════════════════ */

    /* Every text node inside the app */
    p, span, div, label, small, li, h1, h2, h3, h4, h5, h6 {{
        color: {TXT};
    }}

    /* Widget labels */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    .stTextInput > label,
    .stTextArea  > label,
    .stFileUploader > label {{
        color: {TXTS} !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }}

    /* Slider labels */
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {{
        color: {TXTS} !important;
        font-size: 0.85rem !important;
    }}
    /* Slider tick values */
    [data-testid="stSlider"] span {{
        color: {TXTS} !important;
    }}

    /* Checkbox labels */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] p,
    [data-testid="stCheckbox"] span {{
        color: {TXTS} !important;
        font-size: 0.86rem !important;
    }}

    /* File uploader — everything inside */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] div {{
        color: {TXTS} !important;
        font-size: 0.86rem !important;
    }}
    [data-testid="stFileUploader"] section {{
        background: {BGIN} !important;
        border: 2px dashed {BDR} !important;
        border-radius: 10px !important;
        transition: border-color 0.2s, background 0.2s !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: {ACC} !important;
        background: {PALE} !important;
    }}
    /* "Browse files" button */
    [data-testid="stFileUploader"] button {{
        background: {NAV} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
    }}
    /* File name items */
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileName"] {{
        color: {TXT} !important;
        font-size: 0.84rem !important;
    }}

    /* Caption elements */
    .stCaption, .stCaption *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {{
        color: {TXTT} !important;
        font-size: 0.78rem !important;
    }}

    /* Alert / info / warning boxes */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {{
        font-size: 0.87rem !important;
    }}

    /* Text inputs */
    [data-testid="stTextInput"] input {{
        background: {BGIN} !important;
        border: 1.5px solid {BDR} !important;
        border-radius: 9px !important;
        padding: 11px 14px !important;
        font-size: 0.9rem !important;
        color: {TXT} !important;
    }}
    [data-testid="stTextInput"] input::placeholder {{ color: {TXTT} !important; }}
    [data-testid="stTextInput"] input:focus {{
        border-color: {ACC} !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.14) !important;
        outline: none !important;
    }}

    /* Text area */
    [data-testid="stTextArea"] textarea {{
        background: {BGIN} !important;
        border: 1.5px solid {BDR} !important;
        border-radius: 9px !important;
        font-size: 0.85rem !important;
        color: {TXT} !important;
        line-height: 1.65 !important;
    }}

    /* Dividers */
    hr {{
        border: none !important;
        border-top: 1px solid {BDR} !important;
        margin: 14px 0 !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(135deg, {MID} 0%, {ACC} 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 0.58rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        box-shadow: 0 2px 10px rgba(59,130,246,0.28) !important;
        transition: all 0.18s ease !important;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, {ACC} 0%, {MID} 100%) !important;
        box-shadow: 0 4px 18px rgba(59,130,246,0.38) !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button:disabled {{
        background: #c5d4e8 !important;
        color: #8fa3be !important;
        box-shadow: none !important;
        transform: none !important;
    }}
    .stDownloadButton > button {{
        background: {NAV} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 0.58rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        box-shadow: 0 2px 8px rgba(27,58,107,0.22) !important;
        transition: all 0.18s ease !important;
    }}
    .stDownloadButton > button:hover {{
        background: {MID} !important;
        box-shadow: 0 4px 14px rgba(27,58,107,0.32) !important;
        transform: translateY(-1px) !important;
    }}

    /* Spinner */
    .stSpinner > div {{ border-top-color: {ACC} !important; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-thumb {{ background: {BLUE}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACC}; }}

    /* ══════════════════════════════════════════
       CUSTOM COMPONENT STYLES
    ══════════════════════════════════════════ */

    .app-header {{
        background: linear-gradient(135deg, {NAV} 0%, {MID} 100%);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 4px 20px rgba(27,58,107,0.18);
        position: relative;
        overflow: hidden;
    }}
    .app-header::after {{
        content: '';
        position: absolute;
        top: -50px; right: -30px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
    }}
    .app-header-title {{
        color: #ffffff !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.3px;
    }}
    .app-header-sub {{
        color: rgba(255,255,255,0.60) !important;
        font-size: 0.78rem !important;
        margin: 3px 0 0 0 !important;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }}
    .app-header-badge {{
        margin-left: auto;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 50px;
        padding: 5px 15px;
        color: rgba(255,255,255,0.92) !important;
        font-size: 0.76rem !important;
        font-weight: 500;
        z-index: 1;
        white-space: nowrap;
    }}

    .card {{
        background: {BGCD};
        border: 1px solid {BDR};
        border-radius: 14px;
        padding: 22px 24px 24px;
        box-shadow: 0 1px 6px rgba(27,58,107,0.07);
        margin-bottom: 18px;
        transition: box-shadow 0.2s;
    }}
    .card:hover {{ box-shadow: 0 4px 18px rgba(27,58,107,0.11); }}

    .card-header-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        padding-bottom: 14px;
        border-bottom: 1px solid {BDR};
    }}
    .card-icon {{
        width: 34px; height: 34px;
        background: {PALE};
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.95rem;
        flex-shrink: 0;
        border: 1px solid {BLUE};
    }}
    .card-title {{
        color: {TXT} !important;
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }}
    .card-dot {{
        width: 7px; height: 7px;
        border-radius: 50%;
        background: {ACC};
        margin-left: auto;
        flex-shrink: 0;
    }}
    .card-dot-amber {{
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #f59e0b;
        margin-left: auto;
        flex-shrink: 0;
        animation: blink 1.4s ease-in-out infinite;
    }}
    @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.2}} }}

    .sec-label {{
        color: {TXTT} !important;
        font-size: 0.73rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.9px;
        text-transform: uppercase;
        margin: 14px 0 8px 0;
        display: block;
    }}

    .mod-grid {{
        display: flex;
        gap: 8px;
        margin: 12px 0 14px;
    }}
    .mod-chip {{
        flex: 1;
        text-align: center;
        padding: 7px 4px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 700;
    }}
    .mod-ok  {{ background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }}
    .mod-bad {{ background: #fef3c7; color: #78350f; border: 1px solid #fde68a; }}

    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 11px;
        border-radius: 50px;
        font-size: 0.74rem;
        font-weight: 700;
    }}
    .badge-ok   {{ background: #d1fae5; color: #065f46; }}
    .badge-blue {{ background: {BLUE}; color: {NAV}; }}

    .info-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid {BDR};
        font-size: 0.87rem;
    }}
    .info-row:last-child {{ border-bottom: none; }}
    .info-lbl {{ color: {TXTT} !important; font-weight: 500; }}
    .info-val {{ color: {TXT} !important; font-weight: 600; }}

    .stage {{
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 10px 14px;
        border-radius: 9px;
        margin-bottom: 6px;
        font-size: 0.88rem;
        font-weight: 500;
    }}
    .stage-done   {{ background: #f0fdf4; color: #15803d !important; border: 1px solid #bbf7d0; }}
    .stage-done * {{ color: #15803d !important; }}
    .stage-active {{ background: {PALE}; color: {MID} !important; border: 1px solid {BLUE}; }}
    .stage-active * {{ color: {MID} !important; }}
    .stage-wait   {{ background: {BGIN}; color: {TXTT} !important; }}
    .stage-wait * {{ color: {TXTT} !important; }}
    .stage-badge  {{ margin-left: auto; font-size: 0.72rem; font-weight: 700; opacity: 0.85; }}

    .prog-track {{
        height: 7px;
        background: {BLUE};
        border-radius: 50px;
        margin: 14px 0 6px;
        overflow: hidden;
    }}
    .prog-fill {{
        height: 100%;
        background: linear-gradient(90deg, {ACC} 0%, #93c5fd 100%);
        border-radius: 50px;
        transition: width 0.5s ease;
    }}
    .prog-lbl {{
        color: {TXTT} !important;
        font-size: 0.76rem !important;
        font-weight: 600;
        text-align: right;
    }}

    .viz-box {{
        background: {BGIN};
        border: 1.5px dashed {BLUE};
        border-radius: 10px;
        padding: 52px 20px;
        text-align: center;
        line-height: 1.6;
    }}

    .ov-pill {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 11px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }}
    .ov-et  {{ background: #fee2e2; color: #991b1b; }}
    .ov-ed  {{ background: #fef3c7; color: #78350f; }}
    .ov-ncr {{ background: #ede9fe; color: #4c1d95; }}

    .rag-box {{
        background: {BGIN};
        border-left: 3px solid {ACC};
        border-radius: 0 8px 8px 0;
        padding: 13px 16px;
        font-size: 0.88rem;
        color: {TXT} !important;
        line-height: 1.65;
        margin-top: 10px;
    }}
    .rag-box * {{ color: {TXT} !important; }}

    .hint {{
        color: {TXTT} !important;
        font-size: 0.78rem !important;
        line-height: 1.5;
        margin: 6px 0 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── App header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <span style="font-size:2rem;z-index:1;">🧠</span>
        <div style="z-index:1;">
            <p class="app-header-title">Brain MRI Assistant</p>
            <p class="app-header-sub">AI-Powered Neuro-Oncology Analysis Platform</p>
        </div>
        <div class="app-header-badge">🔬 BraTS Pipeline v2</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "app_state": "upload",
    "report_text": "",
    "rag_answer": "",
    "job_id": None,
    "pdf_path": None,
    "pipeline_error": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

STAGE_LABELS = {
    "segmenting": "Tumor Segmentation (nnUNet)",
    "extracting":  "Feature Extraction",
    "generating":  "Report Generation (Gemini)",
    "exporting":   "PDF Export",
}
STAGE_ORDER = ["segmenting", "extracting", "generating", "exporting"]
REQUIRED_MODALITIES = {"t1", "t1ce", "t2", "flair"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _detect_modalities(files):
    ALIASES = {"t1n": "t1", "t1c": "t1ce", "t2w": "t2", "t2f": "flair"}
    found = set()
    for f in files:
        stem = (f.name or "").lower().replace(".nii.gz", "").replace(".nii", "")
        suffix = stem.split("_")[-1] if "_" in stem else stem.split("-")[-1]
        found.add(ALIASES.get(suffix, suffix))
    return found


def display_pdf(file_path: str):
    """Read a PDF from *file_path* and render it in an embedded viewer."""
    if not file_path:
        st.error("No PDF path provided.")
        return
    if not os.path.exists(file_path):
        st.error("PDF file not found.")
        return
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_html = (
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
            f'width="100%" height="700px" '
            f'style="border:1px solid {BDR}; border-radius:8px;" '
            f'type="application/pdf"></iframe>'
        )
        components.html(pdf_html, height=720, scrolling=False)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")


def query_rag(question: str) -> str:
    job_id = st.session_state.job_id
    if not job_id:
        return "No analysis has been run yet."
    try:
        resp = requests.post(
            f"{API_BASE}/api/chat/{job_id}", json={"question": question}, timeout=60
        )
        if resp.status_code == 400:
            return resp.json().get("detail", "Request rejected.")
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except requests.RequestException as exc:
        return f"Error contacting backend: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD STATE
# ═══════════════════════════════════════════════════════════════════════════════
def render_upload_section():
    st.markdown(
        '<div class="card">'
        '<div class="card-header-row">'
        '  <div class="card-icon">📂</div>'
        f'  <p class="card-title">Upload MRI Case Folder</p>'
        '  <div class="card-dot"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Select all NIfTI files from the case folder",
        type=["nii.gz"],
        accept_multiple_files=True,
        key="up_folder",
        help="Must include t1, t1ce, t2, and flair sequences.",
    )

    all_present = False
    if uploaded_files:
        found = _detect_modalities(uploaded_files)
        missing = REQUIRED_MODALITIES - found

        chips = ""
        for mod in sorted(REQUIRED_MODALITIES):
            ok = mod in found
            cls = "mod-ok" if ok else "mod-bad"
            icon = "✓" if ok else "✗"
            chips += f'<div class="mod-chip {cls}">{icon} {mod.upper()}</div>'
        st.markdown(f'<div class="mod-grid">{chips}</div>', unsafe_allow_html=True)

        if missing:
            st.warning(f"Missing modalities: **{', '.join(sorted(m.upper() for m in missing))}**")
        else:
            st.success("All required modalities detected.")
        all_present = not missing
    else:
        st.markdown(
            f'<p class="hint">Required: T1 · T1CE · T2 · FLAIR &nbsp;(.nii.gz)</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    case_id = st.text_input(
        "Case / Patient ID",
        placeholder="e.g. BraTS-GLI-00003-000",
        key="case_id_input",
    )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    ready = all_present and bool(case_id and case_id.strip())

    if st.button("▶  Run Analysis", disabled=not ready, use_container_width=True):
        with st.spinner("Uploading files to backend…"):
            try:
                multipart = [
                    ("files", (f.name, f.getvalue(), "application/octet-stream"))
                    for f in uploaded_files
                ]
                resp = requests.post(
                    f"{API_BASE}/api/analyze",
                    data={"case_id": case_id.strip()},
                    files=multipart,
                    timeout=120,
                )
                resp.raise_for_status()
                st.session_state.job_id = resp.json()["job_id"]
                st.session_state.pipeline_error = ""
                st.session_state.app_state = "loading"
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Info card
    st.markdown(
        f'<div class="card">'
        f'<div class="card-header-row">'
        f'  <div class="card-icon">ℹ️</div>'
        f'  <p class="card-title">Pipeline Overview</p>'
        f'</div>'
        f'<div class="info-row"><span class="info-lbl">Segmentation model</span><span class="info-val">nnUNet v2</span></div>'
        f'<div class="info-row"><span class="info-lbl">Report model</span><span class="info-val">Gemini Pro</span></div>'
        f'<div class="info-row"><span class="info-lbl">Typical duration</span><span class="info-val">3 – 5 min</span></div>'
        f'<div class="info-row"><span class="info-lbl">Output format</span><span class="info-val">PDF + RAG Q&amp;A</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING STATE
# ═══════════════════════════════════════════════════════════════════════════════
def render_loading_section():
    job_id = st.session_state.job_id
    status_data = {"status": "running", "stage": "segmenting",
                   "progress_pct": 0, "error_message": None}
    try:
        resp = requests.get(f"{API_BASE}/api/status/{job_id}", timeout=10)
        resp.raise_for_status()
        status_data = resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach backend: {exc}")

    current_stage = status_data.get("stage", "segmenting")
    progress_pct  = status_data.get("progress_pct", 0)
    status        = status_data.get("status", "running")

    try:
        current_idx = STAGE_ORDER.index(current_stage)
    except ValueError:
        current_idx = -1

    st.markdown(
        '<div class="card">'
        '<div class="card-header-row">'
        '  <div class="card-icon">⏳</div>'
        f'  <p class="card-title">Analyzing MRI Case…</p>'
        '  <div class="card-dot-amber"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    for i, key in enumerate(STAGE_ORDER):
        label = STAGE_LABELS[key]
        if status == "done" or i < current_idx:
            st.markdown(
                f'<div class="stage stage-done">'
                f'  <span>✅</span> {label}'
                f'  <span class="stage-badge">Complete</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif i == current_idx and status == "running":
            st.markdown(
                f'<div class="stage stage-active">'
                f'  <span>⏳</span> {label}'
                f'  <span class="stage-badge">Running…</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="stage stage-wait">'
                f'  <span style="opacity:0.4;">○</span> {label}'
                f'</div>',
                unsafe_allow_html=True,
            )

    pct = min(progress_pct, 100)
    st.markdown(
        f'<div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>'
        f'<p class="prog-lbl">{pct}% complete</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="hint" style="margin-top:10px;">Please keep this tab open. '
        f'Analysis typically completes in 3–5 minutes.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if status == "done":
        try:
            rpt = requests.get(f"{API_BASE}/api/report/{job_id}", timeout=30)
            rpt.raise_for_status()
            st.session_state.report_text = rpt.text
        except requests.RequestException:
            st.session_state.report_text = "(Report could not be fetched.)"
        try:
            pr = requests.get(f"{API_BASE}/api/report/{job_id}/pdf_path", timeout=30)
            st.session_state.pdf_path = (
                pr.json().get("pdf_path") if pr.status_code == 200 else None
            )
        except requests.RequestException:
            st.session_state.pdf_path = None
        st.session_state.app_state = "report"
        st.rerun()

    if status == "error":
        msg = status_data.get("error_message") or "Pipeline encountered an error."
        st.error(msg)
        if st.button("← Back to Upload"):
            st.session_state.app_state = "upload"
            st.rerun()
        return

    time.sleep(4)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT STATE
# ═══════════════════════════════════════════════════════════════════════════════
def render_report_section():
    st.markdown(
        '<div class="card">'
        '<div class="card-header-row">'
        '  <div class="card-icon">📋</div>'
        f'  <p class="card-title">Structured Radiology Report</p>'
        '  <span class="badge badge-ok" style="margin-left:auto;">✓ Generated</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.report_text:
        st.info("No report generated yet.")
    else:
        pdf_path = st.session_state.get("pdf_path")
        if pdf_path:
            display_pdf(pdf_path)
        else:
            st.info("PDF preview unavailable — showing plain text.")
            st.text_area("Report", value=st.session_state.report_text,
                         height=480, disabled=True, label_visibility="collapsed")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("⬇  Download PDF Report", data=f,
                                   file_name=os.path.basename(pdf_path),
                                   mime="application/pdf", use_container_width=True)
        else:
            st.download_button("⬇  Download Report (TXT)",
                               data=st.session_state.report_text.encode(),
                               file_name="report.txt", mime="text/plain",
                               use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # RAG card
    st.markdown(
        '<div class="card">'
        '<div class="card-header-row">'
        '  <div class="card-icon">💬</div>'
        f'  <p class="card-title">Ask About This Case</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Clinical question",
        placeholder="e.g. What regions show enhancement?",
        key="rag_input",
        label_visibility="collapsed",
    )

    if st.button("Ask →", use_container_width=True) and question:
        st.session_state.rag_answer = query_rag(question)

    if st.session_state.rag_answer:
        st.markdown(
            f'<div class="rag-box">{st.session_state.rag_answer}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.button("↺  Start New Case", use_container_width=True):
        st.session_state.update({
            "app_state": "upload", "report_text": "", "rag_answer": "",
            "job_id": None, "pdf_path": None, "pipeline_error": "",
        })
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION PANEL
# ═══════════════════════════════════════════════════════════════════════════════
def render_visualization_panel():
    st.markdown(
        '<div class="card">'
        '<div class="card-header-row">'
        '  <div class="card-icon">🖼</div>'
        f'  <p class="card-title">MRI Visualization Workspace</p>'
        '  <span class="badge badge-blue" style="margin-left:auto;">Preview</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="viz-box">'
        f'  <div style="font-size:2.2rem;margin-bottom:10px;opacity:0.3;">🧠</div>'
        f'  <span style="color:{TXTS};font-weight:500;font-size:0.9rem;">'
        f'  MRI slice rendering will appear here</span><br>'
        f'  <span style="font-size:0.78rem;color:{TXTT};">Supports T1 · T1CE · T2 · FLAIR views</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown(f'<span class="sec-label">Slice Navigation</span>', unsafe_allow_html=True)
    st.slider("Slice Index", 0, 155, 0, disabled=True, key="slice_idx",
              label_visibility="collapsed")

    st.divider()

    st.markdown(f'<span class="sec-label">Tumor Region Overlays</span>', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-bottom:10px;">'
        '<span class="ov-pill ov-et">● Enhancing Tumor</span>'
        '<span class="ov-pill ov-ed">● Edema</span>'
        '<span class="ov-pill ov-ncr">● Necrotic Core</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.checkbox("ET", value=False, disabled=True, key="ov_et")
    c2.checkbox("ED", value=False, disabled=True, key="ov_ed")
    c3.checkbox("NCR", value=False, disabled=True, key="ov_ncr")

    st.divider()

    st.markdown(f'<span class="sec-label">Overlay Transparency</span>', unsafe_allow_html=True)
    st.slider("Transparency", 0, 100, 50, disabled=True, key="ov_alpha",
              label_visibility="collapsed")

    st.divider()

    st.markdown(f'<span class="sec-label">Volume Estimates</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-row"><span class="info-lbl">Enhancing Tumor (ET)</span>'
        '<span class="info-val">— cm³</span></div>'
        '<div class="info-row"><span class="info-lbl">Tumor Core (TC)</span>'
        '<span class="info-val">— cm³</span></div>'
        '<div class="info-row"><span class="info-lbl">Whole Tumor (WT)</span>'
        '<span class="info-val">— cm³</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([9, 11])

with col_left:
    if st.session_state.app_state == "upload":
        render_upload_section()
    elif st.session_state.app_state == "loading":
        render_loading_section()
    elif st.session_state.app_state == "report":
        render_report_section()

with col_right:
    render_visualization_panel()