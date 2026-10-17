"""BriefForge AI — Packaging Brief Assistant."""
import os
import io
import glob
import dotenv
dotenv.load_dotenv()

import streamlit as st
from core.ingest import ingest, Chunk
from core.schema import completeness, FIELD_LABELS, region_hint
from core.llm import extract_with_llm
import core.llm as _llm_mod
from core.gaps import detect_conflicts, order_questions, order_questions_llm, detect_conflicts_llm, apply_answer
from core.brief import render_markdown, export_pdf

BUNDLES = {
    "Esko official (Granola 480g, US-FDA)": "Samples by Esko",
    "Synthetic (Mango Oats Bar, IN-FSSAI)": "samples",
}

st.set_page_config(page_title="BriefForge AI", page_icon="✦", layout="wide")

# ---------------------------------------------------------------- futuristic UI CSS
FUTURISTIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg: #05070a;
  --bg-secondary: #0a0e17;
  --card: rgba(15, 23, 42, 0.6);
  --card-border: rgba(148, 163, 184, 0.08);
  --accent: #22d3ee;
  --accent-glow: rgba(34, 211, 238, 0.35);
  --accent-2: #a855f7;
  --accent-2-glow: rgba(168, 85, 247, 0.3);
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --success: #4ade80;
  --warning: #fbbf24;
  --danger: #f87171;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent;
  padding-top: 1.5rem;
}

[data-testid="stHeader"] { background: transparent; }

/* Animated mesh gradient background layer */
.mesh-bg {
  position: fixed;
  inset: 0;
  z-index: -1;
  background:
    radial-gradient(circle at 15% 20%, rgba(34, 211, 238, 0.15) 0%, transparent 40%),
    radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.12) 0%, transparent 40%),
    radial-gradient(circle at 50% 85%, rgba(59, 130, 246, 0.1) 0%, transparent 45%);
  filter: blur(60px);
  animation: mesh-move 20s ease-in-out infinite alternate;
}

@keyframes mesh-move {
  0% { transform: translate(0, 0); }
  100% { transform: translate(30px, 40px); }
}

/* Glass card */
.glass {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 1.5rem;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
  animation: fadeUp 0.7s ease-out both;
}

.glass:hover {
  border-color: rgba(148, 163, 184, 0.18);
  transform: translateY(-2px);
  transition: all 0.3s ease;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}

.stagger-1 { animation-delay: 0.08s; }
.stagger-2 { animation-delay: 0.16s; }
.stagger-3 { animation-delay: 0.24s; }
.stagger-4 { animation-delay: 0.32s; }

/* Header */
.app-header {
  padding: 1rem 0 2rem 0;
  animation: fadeUp 0.6s ease-out both;
}
.app-title {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0;
  background: linear-gradient(135deg, #fff 0%, #22d3ee 40%, #a855f7 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.app-subtitle {
  color: var(--text-muted);
  font-size: 0.9rem;
  margin-top: 0.35rem;
  font-weight: 400;
}

/* Metric cards */
.metric-row {
  display: flex;
  gap: 1rem;
  margin: 1.5rem 0 2rem 0;
  animation: fadeUp 0.7s ease-out both;
}
.metric-card {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 14px;
  padding: 1.25rem 1rem;
  position: relative;
  overflow: hidden;
}
.metric-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  opacity: 0.9;
}
.metric-value {
  font-size: 1.75rem;
  font-weight: 700;
  line-height: 1.1;
}
.metric-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

/* Section titles */
.section-title {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 2rem 0 1rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text);
}
.section-title .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 12px var(--accent-glow);
}

/* Status badges */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.status-found { background: rgba(74, 222, 128, 0.12); color: var(--success); border: 1px solid rgba(74, 222, 128, 0.25); }
.status-conflict { background: rgba(251, 191, 36, 0.12); color: var(--warning); border: 1px solid rgba(251, 191, 36, 0.25); }
.status-review { background: rgba(59, 130, 246, 0.12); color: var(--accent); border: 1px solid rgba(59, 130, 246, 0.25); }
.status-missing { background: rgba(248, 113, 113, 0.1); color: var(--danger); border: 1px solid rgba(248, 113, 113, 0.2); }

/* Field rows */
.field-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.85rem 1rem;
  background: rgba(2, 6, 23, 0.4);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  margin-bottom: 0.6rem;
  animation: fadeUp 0.5s ease-out both;
}
.field-row:hover {
  border-color: rgba(148, 163, 184, 0.16);
  background: rgba(30, 41, 59, 0.5);
}
.field-key {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  min-width: 150px;
}
.field-value {
  flex: 1;
  font-size: 0.92rem;
  color: var(--text);
  word-break: break-word;
}
.field-source {
  font-size: 0.7rem;
  color: var(--text-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border: none;
  color: #fff;
  font-weight: 600;
  border-radius: 10px;
  padding: 0.6rem 1.2rem;
  box-shadow: 0 0 20px var(--accent-glow);
  transition: all 0.25s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 0 30px var(--accent-glow), 0 0 50px var(--accent-2-glow);
}
.stButton > button:active { transform: translateY(0); }

/* Sidebar styling */
[data-testid="stSidebar"] {
  background: rgba(10, 14, 23, 0.7);
  border-right: 1px solid var(--card-border);
  backdrop-filter: blur(16px);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: var(--text);
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
  color: var(--text-muted);
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Inputs */
.stTextInput > div > div > input {
  background: rgba(2, 6, 23, 0.6);
  border: 1px solid var(--card-border);
  color: var(--text);
  border-radius: 10px;
}
.stTextInput > div > div > input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--accent-glow);
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.15); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.3); }
</style>
"""

st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)
st.markdown('<div class="mesh-bg"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- session state
for k, dv in [("chunks", []), ("state", {}), ("issues", []),
              ("queue", []), ("qi", 0), ("backend", "?"), ("chat", [])]:
    if k not in st.session_state:
        st.session_state[k] = dv

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="glass stagger-1">', unsafe_allow_html=True)
    st.markdown("### Source Files", unsafe_allow_html=True)
    bundle = st.radio("Demo bundle", list(BUNDLES), index=0, label_visibility="collapsed")
    files = st.file_uploader(
        "Upload PDF / DOCX / XLSX / JPG / PNG",
        accept_multiple_files=True,
        type=["pdf", "docx", "xlsx", "xls", "png", "jpg", "jpeg", "txt"],
        label_visibility="collapsed",
    )
    load_samples = st.button("Load bundle files", use_container_width=True)
    if load_samples:
        folder = BUNDLES[bundle]
        st.session_state.chunks = []
        for p in sorted(glob.glob(f"{folder}/*")):
            if os.path.isdir(p):
                continue
            with open(p, "rb") as f:
                st.session_state.chunks += ingest(os.path.basename(p), f.read())
        st.success(f"Loaded {len(st.session_state.chunks)} sections")
    if files:
        chunks: list[Chunk] = []
        for uf in files:
            chunks += ingest(uf.name, uf.read())
        st.session_state.chunks = chunks
        st.success(f"Parsed {len(chunks)} sections from {len(files)} files")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.chunks:
        st.markdown('<div class="glass stagger-2">', unsafe_allow_html=True)
        if st.button("Analyze → build field map", type="primary", use_container_width=True):
            with st.spinner("Extracting…"):
                state, backend = extract_with_llm(st.session_state.chunks)
                issues = detect_conflicts(st.session_state.chunks, state)

                if backend != "offline-rules":
                    with st.spinner("Running AI conflict review…"):
                        llm_issues = detect_conflicts_llm(st.session_state.chunks, state)
                        existing = {(i.field, i.message) for i in issues}
                        for li in llm_issues:
                            key = (li.field, li.message)
                            if key not in existing:
                                issues.append(li)
                                existing.add(key)

                st.session_state.state = state
                st.session_state.issues = issues
                if backend != "offline-rules":
                    st.session_state.queue = order_questions_llm(
                        issues, st.session_state.chunks
                    )
                else:
                    st.session_state.queue = order_questions(issues)
                st.session_state.qi = 0
                st.session_state.backend = backend
                st.session_state.chat = []
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass stagger-3">', unsafe_allow_html=True)
    if st.session_state.backend == "offline-rules" and _llm_mod.last_error:
        st.warning(f"LLM unavailable — using offline rules. {_llm_mod.last_error}")
    st.caption(f"Backend: `{st.session_state.backend}`")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- empty state
if not st.session_state.state:
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">BriefForge AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Upload packaging files → extract structured fields → resolve gaps → export an agency-ready brief.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.info("Load the Esko bundle or upload your own files, then click **Analyze → build field map** in the sidebar.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------- main dashboard
state = st.session_state.state
comp = completeness(state)
region = region_hint(state)

# Header
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">BriefForge AI</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="app-subtitle">Region: {region} &nbsp;·&nbsp; Backend: {st.session_state.backend} &nbsp;·&nbsp; Issues: {len(st.session_state.issues)}</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Metric row
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card stagger-1">
    <div class="metric-label">Completeness</div>
    <div class="metric-value">{comp}%</div>
  </div>
  <div class="metric-card stagger-2">
    <div class="metric-label">Region</div>
    <div class="metric-value">{region}</div>
  </div>
  <div class="metric-card stagger-3">
    <div class="metric-label">Open Issues</div>
    <div class="metric-value">{len(st.session_state.issues)}</div>
  </div>
  <div class="metric-card stagger-4">
    <div class="metric-label">Fields Extracted</div>
    <div class="metric-value">{sum(1 for v in state.values() if v.value)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Progress bar
st.progress(comp / 100)

col1, col2 = st.columns([1.25, 1])

with col1:
    st.markdown('<h3 class="section-title"><span class="dot"></span>Field Map</h3>', unsafe_allow_html=True)
    for key, pf in state.items():
        status_icon = {"found": "●", "missing": "○", "conflict": "▲", "needs_review": "◐"}.get(pf.status, "?")
        status_class = f"status-{pf.status}"
        src = "; ".join(f"{s.file} {s.loc}" for s in pf.sources[:2]) or "—"
        value = (pf.value or "—")[:120]
        conf = pf.confidence.upper()
        st.markdown(f"""
        <div class="field-row stagger-1">
          <span class="field-key">{FIELD_LABELS.get(key, key)}</span>
          <span class="field-value">{value}</span>
          <span class="status-badge {status_class}">{status_icon} {pf.status}</span>
          <span style="font-size:0.7rem; color:var(--text-muted); font-family:monospace;">{conf}</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Source quotes (anti-hallucination evidence)"):
        for c in st.session_state.chunks:
            st.code(f"[{c.file} {c.loc}] {c.text[:400]}")

with col2:
    st.markdown('<h3 class="section-title"><span class="dot"></span>Follow-up Questions</h3>', unsafe_allow_html=True)
    q = st.session_state.queue
    i = st.session_state.qi
    for role, msg in st.session_state.chat:
        st.chat_message(role).write(msg)

    if i < len(q):
        issue = q[i]
        st.markdown(f"""
        <div class="glass">
          <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">
            Question {i+1} of {len(q)}
          </div>
          <div style="font-size:1rem; color:var(--text); margin-bottom:0.75rem;">
            <strong>{FIELD_LABELS.get(issue.field, issue.field)}</strong> — {issue.message}
          </div>
        </div>
        """, unsafe_allow_html=True)
        if issue.evidence:
            st.caption("Evidence: " + " | ".join(issue.evidence[:2]))
        ans = st.text_input("Your answer", key=f"ans_{i}",
                            placeholder="Type your answer…", label_visibility="collapsed")
        if st.button("Submit", key=f"sub_{i}", use_container_width=True):
            if ans.strip():
                apply_answer(state, issue.field, ans.strip())
                st.session_state.chat += [("user", ans.strip())]
                st.session_state.qi += 1
                st.session_state.issues = detect_conflicts(st.session_state.chunks, state)
                st.rerun()
    else:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.success("All blockers resolved — ready to generate the brief.")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.markdown('<h3 class="section-title"><span class="dot"></span>Agency Brief</h3>', unsafe_allow_html=True)
default_md = render_markdown(state)
edited = st.text_area("Review and edit the brief", default_md, height=460, label_visibility="collapsed")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Preview")
    st.markdown(edited)
with c2:
    st.markdown("### Export")
    if st.button("Export agency PDF", type="primary", use_container_width=True):
        path = "Brief_Agency.pdf"
        export_pdf(edited, path)
        with open(path, "rb") as f:
            st.download_button("Download Brief_Agency.pdf", f.read(), file_name="Brief_Agency.pdf")
    st.download_button("Download brief (.md)", edited.encode(), file_name="brief.md", use_container_width=True)
