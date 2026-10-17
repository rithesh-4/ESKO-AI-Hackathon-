"""Gap + conflict detection across sources."""
from __future__ import annotations
import re
from dataclasses import dataclass
from .schema import ProductField, PRIORITY
from .ingest import Chunk


@dataclass
class Issue:
    kind: str  # missing | conflict | review
    field: str
    message: str
    evidence: list[str]
    priority: int  # 0 = blocker


def _all_grams(chunks: list[Chunk]) -> list[Chunk]:
    return chunks


def _resolved(state: dict[str, ProductField], field: str) -> bool:
    """User already answered this field in chat — their answer is source of truth."""
    pf = state.get(field)
    return bool(pf and any(s.file == "user-answer" for s in pf.sources))


def detect_conflicts(chunks: list[Chunk], state: dict[str, ProductField]) -> list[Issue]:
    issues: list[Issue] = []

    # 1. Pack-size / net-qty mismatch: only weights on lines about pack/net/serving/size.
    # Ignores nutrition table grams (per 100g etc.) to avoid false alarms.
    KEYWORDS = ("net", "pack", "serving", "serve", "size", "wt", "quantity", "pouch")
    weights: list[tuple[str, str]] = []
    for c in chunks:
        for line in c.text.splitlines():
            low = line.lower()
            if "per 100" in low or "per serve" in low or low.strip().startswith("nutrient"):
                continue  # nutrition table, not pack size
            if any(k in low for k in KEYWORDS):
                for m in re.finditer(r"(\d+\s?(?:g|ml|kg)\b)", line, re.IGNORECASE):
                    norm = m.group(1).lower().replace(" ", "")
                    weights.append((norm, f"{c.file} {c.loc}: {line.strip()[:120]}"))
    uniq = sorted(set(w for w, _ in weights))
    if len(uniq) > 1 and not _resolved(state, "pack_size"):
        issues.append(Issue("conflict", "pack_size",
            f"Pack size conflict: found {', '.join(uniq)}. Confirm correct net quantity before briefing agency.",
            [e for _, e in weights[:4]], 0))
        if state.get("pack_size"):
            state["pack_size"].status = "conflict"
            state["pack_size"].confidence = "low"

    # 2. Missing mandatory fields (region/certs/tagline optional — never block;
    # serving fields only required for US-FDA bundles, not IN-FSSAI)
    from .schema import region_hint as _rh
    _region = _rh(state)
    for key, pf in state.items():
        if key in ("tagline", "design_notes", "languages", "region", "certifications"):
            continue
        if key in ("serving_size", "servings_per_pack") and _region == "IN-FSSAI":
            continue
        if pf.status in ("missing",) or not pf.value:
            pri = PRIORITY.get(key, 2)
            issues.append(Issue("missing", key,
                f"{pf.label} is missing from all uploaded files.",
                [], pri))

    full = "\n".join(c.text for c in chunks)
    low_full = full.lower()

    # 3. Allergens explicitly unknown vs absent — needs human review, never invent
    if state.get("allergens") and not state["allergens"].value:
        issues.append(Issue("review", "allergens",
            "Allergen declaration not found. Do NOT print without confirming (FDA/FSSAI risk).",
            [], 0))

    # 3b. Allergen advisory wording: "may contain" an item that IS in the ingredient
    # list. Scanned from source chunks (not state values) so it survives user edits.
    if not _resolved(state, "allergens"):
        ing_blob = " ".join(c.text for c in chunks if "ingredient" in c.text.lower()).lower()
        adv_blob = " ".join(c.text for c in chunks if "allergen" in c.text.lower()).lower()
        if "may contain" in adv_blob:
            for nut in ("almond", "pecan", "walnut", "oat"):
                if nut in ing_blob and nut in adv_blob:
                    issues.append(Issue("review", "allergens",
                        f"Allergen box says 'May Contain {nut}' but {nut} is a direct ingredient — confirm advisory vs contains error.",
                        [], 0))
                    break

    # 4. Logo rule vague
    _logo = state.get("branding_logo")
    if _logo and (not _logo.value or _logo.status == "needs_review"):
        issues.append(Issue("review", "branding_logo",
            "Unclear whether existing logo must remain unchanged — ask brand manager.",
            [], 1))

    # 5. Honey % conflict (Esko trap: 1% vs 9%)
    honey_vals: list[tuple[str, str]] = []
    for c in chunks:
        for m in re.finditer(r"honey\s*\((\d+\s?%)\)", c.text, re.I):
            honey_vals.append((m.group(1), f"{c.file} {c.loc}: ...{m.group(0)}..."))
    if len(set(v for v, _ in honey_vals)) > 1 and not _resolved(state, "ingredients"):
        issues.append(Issue("conflict", "ingredients",
            f"Ingredient % conflict: honey {', '.join(sorted(set(v for v, _ in honey_vals)))}. Confirm correct % before briefing.",
            [e for _, e in honey_vals], 0))
        if state.get("ingredients"):
            state["ingredients"].status = "conflict"

    # 6. Vegan claim vs honey (honey is not vegan)
    has_vegan = bool(re.search(r"\bvegan\b", low_full))
    has_honey = "honey" in low_full
    if has_vegan and has_honey and not _resolved(state, "claims"):
        issues.append(Issue("conflict", "claims",
            "Vegan claim conflicts with honey in ingredients (honey is not vegan). Confirm claim or recipe.",
            ["claim: vegan (front panel / content sheet)", "ingredient: honey"], 0))
        if state.get("claims"):
            state["claims"].status = "conflict"

    # 7. Dual nutrition panels (per-100g + FDA) — flag which governs
    has_100g = "per 100g" in low_full or "per100g" in low_full
    has_fda = "nutrition facts" in low_full and "% daily value" in low_full
    if has_100g and has_fda:
        issues.append(Issue("review", "nutrition",
            "Two nutrition formats found (per-100g + US FDA). Confirm FDA panel governs for US market.",
            [], 1))

    # 8. Serving size illegible / unit missing (only when a doubtful value exists;
    # pure missing is already covered by the missing-field question above)
    _ss = state.get("serving_size")
    if _ss and _ss.status == "needs_review" and _ss.value:
        issues.append(Issue("review", "serving_size",
            "Serving size not clearly legible in artwork. Confirm grams from art file.",
            [s.quote[:120] for s in _ss.sources[:2]], 0))
    elif _ss and _ss.value and re.fullmatch(r"\s*\d+\s*", _ss.value):
        issues.append(Issue("review", "serving_size",
            f"Serving size '{_ss.value.strip()}' has no unit. Confirm g/ml.",
            [], 0))
        _ss.status = "needs_review"

    # 9. Ingredient list divergence across sources (dropped items)
    ing_texts = [c.text for c in chunks if "ingredient" in c.text.lower()]
    if len(ing_texts) >= 2 and not _resolved(state, "ingredients"):
        probe = ("walnut", "flax", "strawberr", "pumpkin")
        a, b = ing_texts[0].lower(), ing_texts[1].lower()
        dropped = [p for p in probe if (p in a) != (p in b)]
        if dropped:
            issues.append(Issue("review", "ingredients",
                f"Ingredient lists differ across files (check: {', '.join(dropped)}). Verify word-for-word against native art.",
                [], 1))

    issues.sort(key=lambda i: (i.priority, i.kind))
    return issues


def order_questions(issues: list[Issue], max_q: int = 5) -> list[Issue]:
    # Blockers first, cap at max_q
    return issues[:max_q]


def detect_conflicts_llm(chunks: list[Chunk], state: dict[str, ProductField]) -> list[Issue]:
    """LLM-powered conflict detection — finds semantic issues regex misses.

    Only runs when an LLM key is configured. Returns additional issues
    to merge with deterministic results. Falls back to empty list when
    no LLM backend is available.
    """
    import json
    from .llm import _llm_config, _llm_extract

    if _llm_config() is None:
        return []

    found = {}
    missing = []
    for k, pf in state.items():
        if pf.value:
            found[k] = {
                "value": pf.value,
                "status": pf.status,
                "confidence": pf.confidence,
            }
        elif pf.status == "missing":
            missing.append(k)

    sources_text = "\n\n".join(
        f"[{c.file} {c.loc}] {c.text[:600]}" for c in chunks[:8]
    )

    prompt = (
        "You are a packaging brief QA reviewer. Review extracted fields and source chunks.\n"
        "Find additional issues beyond obvious regex mismatches: semantic contradictions,\n"
        "claims vs ingredient conflicts, vague/TBC values, missing mandatory fields for region,\n"
        "unclear logo rules, serving size without units.\n\n"
        f"EXTRACTED FIELDS:\n{json.dumps(found, indent=2)}\n\n"
        f"MISSING FIELDS:\n{json.dumps(missing, indent=2)}\n\n"
        f"SOURCES:\n{sources_text}\n\n"
        'Return JSON only: {"issues": [{"field": "<key>", "kind": "conflict|missing|review", '
        '"message": "<human readable>", "evidence": ["<verbatim quote>"], "priority": 0|1|2}]}\n'
        "Be conservative. Priority: 0=blocker, 1=review, 2=minor."
    )

    try:
        data = _llm_extract(prompt)
        issues = []
        for item in data.get("issues", []):
            issues.append(
                Issue(
                    kind=str(item.get("kind", "review")),
                    field=str(item.get("field", "general")),
                    message=str(item.get("message", "")),
                    evidence=list(item.get("evidence", []))[:3],
                    priority=int(item.get("priority", 1)),
                )
            )
        return issues
    except Exception:
        return []


def order_questions_llm(
    issues: list[Issue], chunks: list[Chunk], max_q: int = 5
) -> list[Issue]:
    """LLM-powered question prioritization and rephrasing.

    Uses the LLM to turn issues into natural, conversational follow-up
    questions for the brand manager. Falls back to deterministic ordering
    when no LLM key is configured.
    """
    import json
    from .llm import _llm_config, _llm_extract

    if _llm_config() is None or not issues:
        return issues[:max_q]

    issues_summary = "\n".join(
        f"- {i.field}: {i.message} (priority {i.priority}, kind {i.kind})"
        for i in issues
    )

    prompt = (
        "You are a packaging brief assistant. Prioritize and rephrase the following issues\n"
        "into clear, conversational follow-up questions for a brand manager. Keep max 5.\n"
        "Return JSON only: {\"questions\": [{\"field\": \"<key>\", \"question\": \"<natural question>\", \"priority\": 0|1|2}]}\n\n"
        f"ISSUES:\n{issues_summary}\n\n"
        "Prioritize: allergens, net quantity, pack size, mandatory content first (blockers).\n"
        "Make questions specific and actionable. Reference the field being asked about."
    )

    try:
        data = _llm_extract(prompt)
        ordered: list[Issue] = []
        seen: set[str] = set()
        for item in data.get("questions", []):
            f = str(item.get("field", ""))
            q = str(item.get("question", ""))
            if f in seen or not q:
                continue
            seen.add(f)
            for i in issues:
                if i.field == f:
                    i.message = q
                    ordered.append(i)
                    break
            if len(ordered) >= max_q:
                break
        return ordered or issues[:max_q]
    except Exception:
        return issues[:max_q]


def apply_answer(state: dict[str, ProductField], field_key: str, answer: str) -> None:
    pf = state.get(field_key)
    if not pf:
        return
    pf.value = answer.strip()[:500]
    pf.status = "found"
    pf.confidence = "high"
    from .schema import Source
    pf.sources.append(Source("user-answer", "chat", answer.strip()[:160]))
