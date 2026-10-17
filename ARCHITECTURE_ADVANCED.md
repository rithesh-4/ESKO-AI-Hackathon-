# BriefForge AI — Advanced AI-Native Architecture

## Research Foundation

This architecture is informed by the following research:

| Paper | Key Insight | Applied To |
|-------|------------|------------|
| **LayoutLMv3** (ACM MM 2022) | Unified text+image+layout Transformer for document AI; SOTA on form/receipt understanding | Multi-modal document parser for scanned/spec PDFs |
| **Donut** (ECCV 2022) | OCR-free end-to-end document understanding; direct image→structured JSON | Image-based spec sheets without OCR dependency |
| **DocETL** (2025) | Agent-driven pipeline optimization; declarative operators; validation agents | Pipeline orchestration and self-optimizing extraction |
| **HEAR** (ACM MM 2025) | Holistic extraction + agentic reasoning closed-loop; conflict-driven re-evaluation | Cross-modal verification and ambiguity resolution |
| **Multi-Agent Benchmark** (2026) | Reflexive > hierarchical > sequential for extraction accuracy; parallel evaluation reduces hallucination | Multi-agent orchestration with verification loops |
| **CTRAG / RIRAG** (2025-2026) | RAG for regulatory compliance; hybrid dense+sparse retrieval + reranking; obligation coverage | FDA/FSSAI regulatory rule retrieval and compliance checking |
| **SmartIngest** (2025) | Guardrails → Classifier → Extractor → Validator → Router; deterministic routing; guardrail entry node | Security guardrails and guardrail-driven routing |
| **Enterprise Agentic AI** | Extractor → Analyst → Verifier with policy engine and audit logging | Separation of concerns and auditability |

## Advanced Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT LAYER                                │
│  PDF/DOCX/XLSX/JPG/PNG uploads  →  Regulatory rules (FDA/FSSAI)         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: MULTI-MODAL INGESTION                       │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────────────┐   │
│  │ Text Parser │  │ LayoutLMv3  │  │ Donut (OCR-free VDU)          │   │
│  │ (PyMuPDF,   │  │ (text+img+  │  │ (image→JSON, no OCR needed)   │   │
│  │  python-docx│  │  layout+OCR)│  │                               │   │
│  └─────────────┘  └─────────────┘  └───────────────────────────────┘   │
│                                  │                                      │
│                          Chunks with layout + provenance                │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 LAYER 2: MULTI-AGENT EXTRACTION PIPELINE                │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────┐       │
│  │  Classifier  │───▶│   Extractor  │───▶│   Verifier Agent    │       │
│  │   Agent      │    │   Agent      │    │  (grounding, conf.) │       │
│  └──────────────┘    └──────────────┘    └──────────┬──────────┘       │
│                                                     │                    │
│                              ┌──────────────────────┴───────────────────┐ │
│                              ▼                                            │ │
│                      ┌──────────────┐    ┌─────────────────────┐         │ │
│                      │ Conflict     │◀───│   Router Agent      │         │ │
│                      │ Resolver     │    │ (approve/flag/reject)│         │ │
│                      │ Agent        │    └─────────────────────┘         │ │
│                      └──────────────┘                                      │ │
│                                                      ┌───────────────────┘ │
│                                                      ▼                      │
│                                              ┌─────────────┐              │ │
│                                              │ Reflexive   │              │ │
│                                              │ Self-Correct│              │ │
│                                              │ Loop (≤3 it)│              │ │
│                                              └─────────────┘              │ │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 LAYER 3: RAG COMPLIANCE ENGINE                          │
│  Hybrid Retrieval (dense BM25 + semantic) + Cross-Encoder Reranking     │
│  FDA/FSSAI regulatory rules → obligation extraction → compliance check   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: HUMAN-IN-THE-LOOP Q&A                       │
│  LLM-generated prioritized questions → user answers → state update     │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LAYER 5: AGENCY BRIEF GENERATION                     │
│  Region-aware brief with citations, confidence scores, audit trail      │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     AUDIT & OBSERVABILITY LAYER                         │
│  Every extraction logged: input, output, confidence, agent, decision    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Model Selection

| Component | Production Model | Fallback (Offline) | Reason |
|-----------|-----------------|-------------------|--------|
| Document parsing | LayoutLMv3-base | PyMuPDF + regex | Best layout+text understanding |
| Image-only docs | Donut (base) | Pillow + placeholder | OCR-free, end-to-end |
| Field extraction | GPT-4o / Claude-3.5 / Nemotron | Regex patterns | Structured JSON with quotes |
| Verification | Smaller model (Haiku / Groq) | Deterministic rules | Fast, cheap verification |
| Compliance RAG | bge-m3 + bge-reranker | BM25 + keyword | Hybrid retrieval + reranking |
| Question gen | LLM (any) | Deterministic ordering | Natural follow-ups |

## Key Innovations vs. Current Solution

1. **OCR-free visual understanding** (Donut) — scanned spec sheets don't need Tesseract
2. **Multi-agent verification loop** — extractor → verifier → corrector, not single-shot
3. **Regulatory RAG** — FDA/FSSAI rules retrieved and checked against extracted fields
4. **Conflict-driven re-evaluation** — ambiguous fields trigger re-extraction (HEAR pattern)
5. **Confidence-calibrated routing** — low-confidence fields auto-flagged for human review
6. **Full audit trail** — every agent decision logged with inputs/outputs
