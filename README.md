# BriefForge AI — Esko AIForge Hackathon (Problem 1)

AI assistant that turns messy product files into a trusted, agency-ready packaging brief.

## Problem
Brand managers start with scattered files (old pack, spec PDF, Excel). Information is
incomplete, inconsistent, hard for designers to use.

## Approach
1. **Ingest** PDF/DOCX/XLSX/JPG with source tracking (`file + page/sheet` + verbatim quote).
   Dieline guide text (`Art orientation`, `Copy free`) is ignored — never treated as content.
2. **Normalize** to a fixed Product Schema (12 mandatory + serving/region/certifications).
   Region auto-detected: `US-FDA` vs `IN-FSSAI`, brief printer rules adapt.
3. **Gap engine**: detects `found / missing / conflict / needs_review` + prioritizes blockers.
   Esko bundle traps caught: honey **1% vs 9%**, vegan-claim-vs-honey, dual nutrition panels,
   illegible serving size, ingredient-list divergence, `May Contain` advisory wording.
   Synthetic bundle trap: spec **45g** vs nutrition sheet **50g**, missing allergens.
4. **Q&A (max 5)**: asks only blockers first. Allergens/serving-size never guessed.
5. **Brief**: region-aware editable agency template with citation chips + PDF export.

## Architecture
```
Uploads → ingest.py (chunks w/ provenance) → llm.py (Gemini/Groq, offline-rules fallback)
→ gaps.py (conflict/missing) → Streamlit Q&A → brief.py (markdown + PDF)
```
See `assets/arch.md` for flow diagram.

## AI models / tech
- LLM: any OpenAI-compatible provider via `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL`
  (stdlib `urllib`, no SDK needed). Free-key picks that fit: **Groq** (fast,
  `llama-3.3-70b-versatile`), **Google AI Studio** (`gemini-2.0-flash`),
  **Cerebras** (fastest), **OpenRouter** (`:free` models) — see `.env.example`.
  NVIDIA NIM also works (`NVIDIA_API_KEY` alias) but account entitlements vary:
  Ultra times out on restricted keys, `llama-3.3-70b-instruct` is retired (410).
  Reasoning disabled for extraction; any backend failure falls back to offline rules.
- Offline: deterministic regex extractor (`offline-rules`) — demo works with no key
- Parsing: PyMuPDF, python-docx, openpyxl, Pillow; export via fpdf2
- Anti-hallucination: LLM quotes must match source verbatim or field drops to `needs_review`

## Run
```bash
pip install -r requirements.txt
python make_samples.py   # regenerates samples/ (synthetic IN-FSSAI bundle)
python -m streamlit run app.py
```
Open the URL, pick a bundle (Esko official default) → **Or load bundle files** → **Analyze**.

With a key (e.g. free Groq key — fastest path to live LLM):
```bash
copy .env.example .env   # uncomment one provider block, add key
```

## Demo (3 min, Esko bundle)
0:00 Load `Samples by Esko` bundle → Analyze (5 issues: honey 1%/9%, vegan-honey, allergen wording, serving, ingredient diff)
0:45 Answer 2–3 questions (honey = 1%; serving = 60g; confirm vegan claim dropped)
1:30 Show provenance table + region badge `US-FDA` + source quotes
2:00 Edit brief → Export `Brief_Agency.pdf` → open PDF
(Alternate: synthetic bundle shows 45g-vs-50g conflict + missing allergens.)

## Submission contents
- Working prototype (`app.py` + `core/`)
- Esko official bundle (`Samples by Esko/`) + synthetic bundle (`samples/`)
- This README + `assets/arch.md` + demo recording (record with OBS/Zoom)
- Repo < 50 MB
