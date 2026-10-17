"""BriefForge AI — Streamlit app (Esko AIForge Problem 1)."""
import os
import io
import dotenv
dotenv.load_dotenv()

import streamlit as st
from core.ingest import ingest, Chunk
from core.schema import completeness, FIELD_LABELS, region_hint
from core.llm import extract_with_llm
import core.llm as _llm_mod  # for last_error (read AFTER extract returns)
from core.gaps import detect_conflicts, order_questions, order_questions_llm, detect_conflicts_llm, apply_answer
from core.brief import render_markdown, export_pdf

BUNDLES = {
    "Esko official (Granola 480g, US-FDA)": "Samples by Esko",
    "Synthetic (Mango Oats Bar, IN-FSSAI)": "samples",
}

st.set_page_config(page_title="BriefForge AI — Packaging Brief Assistant", layout="wide")
st.title("BriefForge AI — From Messy Files to Agency-Ready Brief")
st.caption("Esko AIForge · Problem 1 · Upload → gaps → 5 smart questions → trusted brief with provenance")

# session state
for k, dv in [("chunks", []), ("state", {}), ("issues", []),
              ("queue", []), ("qi", 0), ("backend", "?"), ("chat", [])]:
    if k not in st.session_state:
        st.session_state[k] = dv

with st.sidebar:
    st.header("1 · Upload source files")
    bundle = st.radio("Demo bundle", list(BUNDLES), index=0)
    files = st.file_uploader("PDF / DOCX / XLSX / JPG / PNG",
                             accept_multiple_files=True,
                             type=["pdf", "docx", "xlsx", "xls", "png", "jpg", "jpeg", "txt"])
    load_samples = st.button("Or load bundle files")
    if load_samples:
        import glob
        folder = BUNDLES[bundle]
        st.session_state.chunks = []
        for p in sorted(glob.glob(f"{folder}/*")):
            if os.path.isdir(p):
                continue
            with open(p, "rb") as f:
                st.session_state.chunks += ingest(os.path.basename(p), f.read())
        st.success(f"Loaded {len(st.session_state.chunks)} chunks from {folder}/")
    if files:
        chunks: list[Chunk] = []
        for uf in files:
            chunks += ingest(uf.name, uf.read())
        st.session_state.chunks = chunks
        st.success(f"Parsed {len(chunks)} sections from {len(files)} files")
    if st.session_state.chunks:
        if st.button("Analyze → build field map", type="primary"):
            with st.spinner("Extracting + checking gaps…"):
                state, backend = extract_with_llm(st.session_state.chunks)
                issues = detect_conflicts(st.session_state.chunks, state)

                # LLM-powered additional conflict detection when live backend available
                if backend != "offline-rules":
                    with st.spinner("Running AI-powered conflict review…"):
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
    st.divider()
    if st.session_state.backend == "offline-rules" and _llm_mod.last_error:
        st.warning(f"LLM unavailable ({_llm_mod.last_error}) — using offline rules. Fix key/network to go live.")
    st.caption(f"AI backend: `{st.session_state.backend}` (offline-rules works with no key; set LLM_API_KEY for live LLM — Groq/Gemini/Cerebras/OpenRouter/NVIDIA, see .env.example)")

# main
if not st.session_state.state:
    st.info("Load a bundle (Esko official default) or upload files → click **Analyze**. Synthetic bundle: `python make_samples.py`.")
    with st.expander("What makes this win?"):
        st.write("- Every field shows its source file + quote (provenance).\n- Conflicts block progress (Mango 45g vs 50g; Granola honey 1% vs 9%, vegan-vs-honey).\n- Max 5 prioritized questions, allergens/serving-size never guessed.\n- Region-aware brief (US-FDA / IN-FSSAI) + editable + PDF export.")
    st.stop()

state = st.session_state.state
comp = completeness(state)
region = region_hint(state)
st.progress(comp / 100, f"Brief completeness: {comp}%  · {region} · ({st.session_state.backend})")

col1, col2 = st.columns([1.15, 1])

with col1:
    st.subheader("Field map with evidence")
    rows = []
    for key, pf in state.items():
        status_icon = {"found": "✅", "missing": "❌", "conflict": "⚠️", "needs_review": "🔍"}.get(pf.status, "?")
        src = "; ".join(f"{s.file} {s.loc}" for s in pf.sources[:2]) or "—"
        rows.append({"Field": FIELD_LABELS.get(key, key), "Value": (pf.value or "—")[:90],
                     "Status": f"{status_icon} {pf.status}", "Source": src, "Conf": pf.confidence})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Source quotes (anti-hallucination evidence)"):
        for c in st.session_state.chunks:
            st.code(f"[{c.file} {c.loc}] {c.text[:400]}")

with col2:
    st.subheader("Smart follow-ups (max 5)")
    q = st.session_state.queue
    i = st.session_state.qi
    for role, msg in st.session_state.chat:
        st.chat_message(role).write(msg)
    if i < len(q):
        issue = q[i]
        st.warning(f"Q{i+1}/{len(q)} · **{FIELD_LABELS.get(issue.field, issue.field)}** — {issue.message}")
        if issue.evidence:
            st.caption("Evidence: " + " | ".join(issue.evidence[:2]))
        ans = st.text_input("Your answer", key=f"ans_{i}", placeholder="e.g. 45g is correct; contains milk, cashew, oats gluten…")
        if st.button("Submit answer", key=f"sub_{i}"):
            if ans.strip():
                apply_answer(state, issue.field, ans.strip())
                st.session_state.chat += [("user", ans.strip())]
                st.session_state.qi += 1
                # re-check conflicts after answer
                st.session_state.issues = detect_conflicts(st.session_state.chunks, state)
                st.rerun()
    else:
        st.success("All blockers resolved — ready to generate the brief.")

st.divider()
st.subheader("Final brief — review & edit before export")
default_md = render_markdown(state)
edited = st.text_area("Edit freely (agency will receive exactly this)", default_md, height=420)
c1, c2 = st.columns(2)
with c1:
    st.markdown(edited)
with c2:
    if st.button("Export agency PDF", type="primary"):
        path = "Brief_Agency.pdf"
        export_pdf(edited, path)
        with open(path, "rb") as f:
            st.download_button("Download Brief_Agency.pdf", f.read(), file_name=path)
    st.download_button("Download brief (.md)", edited.encode(), file_name="brief.md")
