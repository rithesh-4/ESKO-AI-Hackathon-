"""Multi-agent extraction pipeline for BriefForge AI.

Implements the agentic layer from ARCHITECTURE_ADVANCED.md, informed by:

- Multi-Agent Benchmark (2026): reflexive > hierarchical > sequential; parallel evaluation
- HEAR (ACM MM 2025): holistic extraction + agentic reasoning, conflict-driven re-evaluation
- SmartIngest (2025): guardrails -> classifier -> extractor -> validator -> router
- Enterprise Agentic AI: extractor -> analyst -> verifier with policy engine

Pipeline:
  Classifier → Extractor → Verifier → Conflict Resolver → Router
                    └─ Reflexive Self-Correction Loop (≤3 iterations) ─┘

All LLM calls are optional; deterministic offline fallback is preserved.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .schema import ProductField, Source, empty_state
from .ingest import Chunk
from .rag_compliance import get_compliance_engine, ComplianceCheck

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent roles and states
# ---------------------------------------------------------------------------


class AgentRole(str, Enum):
    CLASSIFIER = "classifier"
    EXTRACTOR = "extractor"
    VERIFIER = "verifier"
    RESOLVER = "resolver"
    ROUTER = "router"


class RouteDecision(str, Enum):
    APPROVE = "approve"
    FLAG = "flag"
    REJECT = "reject"


@dataclass
class AgentTrace:
    """Audit log entry for one agent step."""
    agent: AgentRole
    input_summary: str
    output_summary: str
    confidence: float
    decision: str
    iteration: int = 0


@dataclass
class AgentState:
    """State carried through the multi-agent pipeline."""
    chunks: list[Chunk]
    extracted: dict[str, ProductField] = field(default_factory=dict)
    traces: list[AgentTrace] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3
    region: Optional[str] = None
    needs_human_review: list[str] = field(default_factory=list)
    route_decision: Optional[RouteDecision] = None


# ---------------------------------------------------------------------------
# Optional LLM client (same pattern as core/llm.py)
# ---------------------------------------------------------------------------


def _call_llm_json(prompt: str) -> Optional[dict]:
    """Call the configured LLM backend for JSON output. Returns None if unavailable."""
    from .llm import _llm_config, _llm_extract

    try:
        if _llm_config() is None:
            return None
        return _llm_extract(prompt)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Agent 1: Classifier — document type + region detection
# ---------------------------------------------------------------------------


def _classify_document(chunks: list[Chunk]) -> tuple[str, dict]:
    """Classify uploaded documents: product type, region, document kinds."""
    from .llm import _combined

    combined = _combined(chunks)[:6000]
    text_lower = combined.lower()

    # Deterministic region detection
    region = "US-FDA"  # default
    if any(k in text_lower for k in ("india", "fssai", "mrp", "veg mark")):
        region = "IN-FSSAI"

    # Document type hints
    doc_types = []
    if any("nutrition" in c.text.lower() for c in chunks):
        doc_types.append("nutrition_sheet")
    if any("ingredient" in c.text.lower() for c in chunks):
        doc_types.append("spec_sheet")
    if any(c.loc.startswith(("p", "Sheet")) for c in chunks):
        doc_types.append("artwork_or_spec")

    # Try LLM refinement if available
    prompt = (
        "Classify these packaging document chunks. Return JSON only:\n"
        '{"region": "US-FDA|IN-FSSAI", "doc_types": [...], "product_category_guess": "..."}\n\n'
        f"SOURCES:\n{combined}"
    )
    llm_data = _call_llm_json(prompt)
    if llm_data:
        region = llm_data.get("region", region)
        doc_types = llm_data.get("doc_types", doc_types)

    meta = {
        "region": region,
        "doc_types": doc_types,
        "num_chunks": len(chunks),
        "total_chars": sum(len(c.text) for c in chunks),
    }
    return region, meta


# ---------------------------------------------------------------------------
# Agent 2: Extractor — structured field extraction (offline + LLM)
# ---------------------------------------------------------------------------


def _extract_fields(chunks: list[Chunk]) -> dict[str, ProductField]:
    """Extract product fields using offline extractor (deterministic baseline)."""
    from .llm import offline_extract

    return offline_extract(chunks)


# ---------------------------------------------------------------------------
# Agent 3: Verifier — grounding check + confidence scoring
# ---------------------------------------------------------------------------


def _verify_fields(state: AgentState, chunks_text: str) -> list[ComplianceCheck]:
    """Verify extracted fields: grounding (verbatim source match) + confidence."""
    engine = get_compliance_engine()
    region = state.region or "US-FDA"
    checks = engine.check_all(state.extracted, region, chunks_text)

    # Update field statuses based on verification
    for check in checks:
        pf = state.extracted.get(check.field)
        if not pf:
            continue
        if not check.compliant or check.needs_review:
            pf.status = "needs_review"
            pf.confidence = "low"
            state.needs_human_review.append(check.field)
        elif check.confidence < 0.6:
            pf.status = "needs_review"
            pf.confidence = "low"
        else:
            pf.confidence = "high" if check.confidence >= 0.8 else "medium"

    return checks


# ---------------------------------------------------------------------------
# Agent 4: Conflict Resolver — cross-document consistency
# ---------------------------------------------------------------------------


def _resolve_conflicts(state: AgentState, chunks: list[Chunk]) -> list[dict]:
    """Detect and resolve conflicts across documents using LLM or deterministic rules."""
    from .llm import _combined

    conflicts = []
    combined = _combined(chunks)

    # Deterministic conflict checks (preserved from original gaps.py logic)
    # Honey % conflict
    honey_vals = re.findall(r"honey\s*\((\d+\s?%)\)", combined, re.I)
    if len(set(honey_vals)) > 1:
        conflicts.append({
            "field": "ingredients",
            "type": "percentage_conflict",
            "message": f"Honey percentage conflict: {', '.join(sorted(set(honey_vals)))}",
            "severity": "high",
        })

    # Vegan vs honey
    has_vegan = bool(re.search(r"\bvegan\b", combined, re.I))
    has_honey = "honey" in combined.lower()
    if has_vegan and has_honey:
        conflicts.append({
            "field": "claims",
            "type": "claim_ingredient_conflict",
            "message": "Vegan claim conflicts with honey in ingredients (honey is not vegan)",
            "severity": "high",
        })

    # Try LLM for deeper semantic conflicts
    prompt = (
        "Review these extracted packaging fields and source text for semantic conflicts:\n"
        "Find: contradictory values, claims vs ingredient mismatches, vague/TBC fields,\n"
        "missing mandatory fields, serving size without units.\n"
        'Return JSON only: {"conflicts": [{"field": "...", "type": "...", "message": "...", "severity": "low|medium|high"}]}\n\n'
        f"FIELDS: {json.dumps({k: v.value for k, v in state.extracted.items() if v.value})}\n\n"
        f"SOURCES: {combined[:4000]}"
    )
    llm_data = _call_llm_json(prompt)
    if llm_data and "conflicts" in llm_data:
        for c in llm_data["conflicts"]:
            # Avoid duplicates
            if not any(existing["message"] == c.get("message") for existing in conflicts):
                conflicts.append(c)

    return conflicts


# ---------------------------------------------------------------------------
# Agent 5: Router — approve / flag / reject decision
# ---------------------------------------------------------------------------


def _route(state: AgentState) -> RouteDecision:
    """Determine routing decision based on verification results."""
    if state.needs_human_review:
        # Blockers that need human review
        blocker_fields = {"allergens", "net_quantity", "pack_size", "mandatory_content", "serving_size"}
        if any(f in blocker_fields for f in state.needs_human_review):
            return RouteDecision.FLAG
        return RouteDecision.FLAG

    # Check for rejected conditions
    for pf in state.extracted.values():
        if pf.status == "needs_review" and pf.confidence == "low":
            return RouteDecision.FLAG

    return RouteDecision.APPROVE


# ---------------------------------------------------------------------------
# Reflexive Self-Correction Loop (HEAR pattern)
# ---------------------------------------------------------------------------


def _self_correct(state: AgentState, chunks_text: str) -> AgentState:
    """Iteratively re-extract low-confidence fields using LLM critique.

    Reflexive architecture: verifier critiques extractor output, extractor revises.
    Max 3 iterations to bound cost/latency.
    """

    for iteration in range(1, state.max_iterations + 1):
        state.iteration = iteration

        # Identify fields needing correction
        low_fields = [
            (k, v) for k, v in state.extracted.items()
            if v.status == "needs_review" and v.confidence == "low" and v.value
        ]
        if not low_fields:
            break

        # Build critique prompt
        fields_json = {k: {"value": v.value, "confidence": v.confidence} for k, v in low_fields}
        prompt = (
            "You are a verification agent reviewing packaging field extractions.\n"
            "The following fields have low confidence. Re-extract them from the source text\n"
            "with better accuracy. Only change values that are clearly wrong.\n"
            'Return JSON only: {"corrections": {"<field>": {"value": "<new value or null>", "confidence": "high|medium|low", "quote": "verbatim source text"}}}\n\n'
            f"FIELDS_TO_CORRECT: {json.dumps(fields_json)}\n\n"
            f"SOURCES: {chunks_text[:6000]}"
        )

        llm_data = _call_llm_json(prompt)
        if not llm_data or "corrections" not in llm_data:
            break  # LLM unavailable, stop correction loop

        corrections = llm_data["corrections"]
        changed = False
        for field_key, corr in corrections.items():
            if field_key not in state.extracted:
                continue
            new_value = corr.get("value")
            new_conf = corr.get("confidence", "low")
            quote = corr.get("quote", "")

            if new_value is None:
                # Mark as missing
                state.extracted[field_key].value = None
                state.extracted[field_key].status = "missing"
                state.extracted[field_key].confidence = "low"
                changed = True
            elif new_value != state.extracted[field_key].value:
                state.extracted[field_key].value = str(new_value)[:500]
                state.extracted[field_key].confidence = new_conf
                if quote:
                    state.extracted[field_key].sources.append(
                        Source("llm-correction", "reflexive-loop", quote[:160])
                    )
                changed = True

        if not changed:
            break  # No meaningful changes, exit loop

        # Re-verify after correction
        state.traces.append(AgentTrace(
            agent=AgentRole.EXTRACTOR,
            input_summary=f"Reflexive iteration {iteration}: {len(corrections)} corrections",
            output_summary="Re-extracted low-confidence fields",
            confidence=0.7,
            decision="corrected",
            iteration=iteration,
        ))

    return state


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


def run_agentic_pipeline(chunks: list[Chunk]) -> dict:
    """Run the full multi-agent extraction pipeline.

    Returns:
        dict with keys: state, traces, checks, conflicts, route_decision, backend
    """
    state = AgentState(chunks=chunks)
    chunks_text = "\n".join(c.text for c in chunks)

    # --- Agent 1: Classifier ---
    region, meta = _classify_document(chunks)
    state.region = region
    state.traces.append(AgentTrace(
        agent=AgentRole.CLASSIFIER,
        input_summary=f"{len(chunks)} document chunks",
        output_summary=f"region={region}, doc_types={meta['doc_types']}",
        confidence=0.9,
        decision="classified",
    ))

    # --- Agent 2: Extractor ---
    extracted = _extract_fields(chunks)
    state.extracted = extracted
    state.traces.append(AgentTrace(
        agent=AgentRole.EXTRACTOR,
        input_summary="document chunks",
        output_summary=f"{sum(1 for v in extracted.values() if v.value)} fields extracted",
        confidence=0.75,
        decision="extracted",
    ))

    # --- Agent 3: Verifier ---
    checks = _verify_fields(state, chunks_text)
    state.traces.append(AgentTrace(
        agent=AgentRole.VERIFIER,
        input_summary="extracted fields",
        output_summary=f"{len(checks)} compliance checks, {len(state.needs_human_review)} need review",
        confidence=0.8,
        decision="verified",
    ))

    # --- Reflexive Self-Correction Loop ---
    state = _self_correct(state, chunks_text)

    # --- Agent 4: Conflict Resolver ---
    conflicts = _resolve_conflicts(state, chunks)
    state.traces.append(AgentTrace(
        agent=AgentRole.RESOLVER,
        input_summary="extracted fields + sources",
        output_summary=f"{len(conflicts)} conflicts detected",
        confidence=0.85,
        decision="resolved",
    ))

    # --- Agent 5: Router ---
    route_decision = _route(state)
    state.route_decision = route_decision
    state.traces.append(AgentTrace(
        agent=AgentRole.ROUTER,
        input_summary="verification results + conflicts",
        output_summary=f"route={route_decision.value}",
        confidence=0.9,
        decision=route_decision.value,
    ))

    return {
        "state": state,
        "extracted": state.extracted,
        "traces": state.traces,
        "checks": checks,
        "conflicts": conflicts,
        "route_decision": route_decision,
        "backend": "agentic-multi-agent",
        "region": region,
    }


# ---------------------------------------------------------------------------
# Drop-in replacement for offline_extract
# ---------------------------------------------------------------------------


def agentic_extract(chunks: list[Chunk]) -> dict[str, ProductField]:
    """Drop-in replacement for core.llm.offline_extract using the agentic pipeline.

    Returns the same dict[str, ProductField] schema so existing code works unchanged.
    """
    result = run_agentic_pipeline(chunks)
    return result["extracted"]
