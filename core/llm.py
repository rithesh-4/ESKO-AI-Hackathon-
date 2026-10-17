"""LLM wrapper: any OpenAI-compatible backend, with offline rule-based fallback.

Configure with generic env (stdlib only, no SDK deps):
  LLM_API_KEY   — provider key (falls back to NVIDIA_API_KEY for back-compat)
  LLM_MODEL     — model id (falls back to NVIDIA_MODEL, default nvidia/nemotron-3-ultra-550b-a55b)
  LLM_BASE_URL  — endpoint root (falls back to NVIDIA_BASE_URL,
                  default https://integrate.api.nvidia.com/v1)
  LLM_TIMEOUT   — seconds (default 120; falls back to NVIDIA_TIMEOUT)

Free-key providers that fit this interface (all OpenAI-compatible
/chat/completions): Groq, Google AI Studio (Gemini), Cerebras, OpenRouter,
Together, Mistral, DeepSeek. See .env.example for endpoints.
Reasoning traces disabled (enable_thinking=false) + stripped before JSON parse:
this task is a single JSON extraction, traces only add latency + parse risk.
All LLM outputs are validated: quotes must appear in source text
(anti-hallucination) or the field drops to needs_review.
"""
from __future__ import annotations
import os
import re
import json
from .schema import empty_state, ProductField, Source
from .ingest import Chunk
from .agents import agentic_extract, run_agentic_pipeline


def _combined(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.file} {c.loc}] {c.text}" for c in chunks)


def _find_quote(chunks: list[Chunk], needle: str, maxlen: int = 160) -> Source | None:
    if not needle:
        return None
    nl = needle.strip().lower()[:60]
    for c in chunks:
        low = c.text.lower()
        idx = low.find(nl)
        if idx != -1:
            s = max(0, idx - 40)
            return Source(c.file, c.loc, c.text[s:s + maxlen].strip())
    return None


# ---------------- offline extractor (regex/keyword, deterministic) ----------------
# Dieline guide marks are never content — strip before matching.
_DIELINE = re.compile(r"\b(art orientation|copy free|front panel orientation|back panel orientation|bottom panel orientation|principal display panel)\b", re.I)

_PATTERNS = {
    "product_name": [r"product\s*name\s*[:\-]\s*(.+)", r"(granola\s+berry\s+crunch)", r"^(.{3,60}bar|.{3,60}biscuit|.{3,60}shampoo)"],
    "pack_size": [r"(?:package size\s*/\s*net weight|portions selected|net\s*(?:wt|qty|quantity|weight)|pack size)\s*[:\|\-]*\s*(\d+\s?(?:g|ml|kg)\b)", r"(\d+\s?g\b|\d+\s?ml\b|\d+\s?kg\b)"],
    "net_quantity": [r"(?:package size\s*/\s*net weight|portions selected|net\s*(?:wt|qty|quantity|weight))\s*[:\|\-]*\s*(\d+\s?(?:g|ml|kg)\b[^\n|]*)"],
    "ingredients": [r"ingredients?\s*[:\-|]\s*([^\n]+(?:\n[^\n]+)?)", r"ingredient\s*list[^\n]*\n([^\n]+(?:\n[^\n]+){0,4})"],
    "allergens": [r"allergen[^:\n]*[:\-]\s*([^\n]+)"],
    "nutrition": [r"^(?=.*\b(?:energy|protein|carbohydrate|fat|sugars?|sodium|calories|% daily value|nutrition)\b)(?=.*(?:\d|g\b|mg\b|kg\b|%|per\s+serving|daily\s+value|%dv)).*$|^.*\bnutrition\s+facts\b.*$"],
    "claims": [r"(?:claim[^:\n]*[:\-]\s*[^\n]+|no added sugar|high fibre|no added refined sugar|source of fibre|oven\s*baked|gmo\s*free)"],
    "certifications": [r"(usda\s*organic|certified\s*vegan|how2recycle|recyclable[^\n]*|non[\s-]?gmo[^\n]*)"],
    "branding_logo": [r"(logo[^:\n]*[:\-]\s*[^\n]+|keep existing[^\n]*logo[^\n]*|logo[^\n]*unchanged|produced by[^\n]+|packaging powered by[^\n]+|logo[^.\n]*\b(?:stay|keep|retain|unchanged|replace|redesign)\b[^.\n]*)"],
    "mandatory_content": [r"(fssai|veg mark|mrp|mfg\.?|expiry|customer care|mfd by|nutrition facts|% daily value|upc\s*/\s*barcode|store in a cool)[^\n]*"],
    "pack_format": [r"(?:package type|pack\s*(?:format|type))\s*[:\|\-]+\s*([a-z]+)", r"(?<!allergen )(pouch|carton|bottle|jar|sachet|packet|box)"],
    "target_market": [r"(united states|north america|india|country of market[^\n]*|market\s*[:\-]\s*[^\n]+|pan-india)"],
    "region": [r"(united states|north america|u\.s\.?|india)"],
    "servings_per_pack": [r"(\d+\s*servings?\s*per\s*(?:container|package|pack))"],
    "serving_size": [r"serving size\s*([^\n]+)"],
    "product_category": [r"(snacks?|beverage|oats|granola|dairy|personal care|shampoo|high-grain)"],
}


def _clean_text(t: str) -> str:
    return _DIELINE.sub(" ", t)


_MULTI_KEYS = ("claims", "certifications", "mandatory_content")


def _collect_multi(key: str, pats: list[str], chunks: list[Chunk]) -> tuple[str, Source] | None:
    seen: list[str] = []
    first_src: Source | None = None
    for c in chunks:
        text = _clean_text(c.text)
        for p in pats:
            for m in re.finditer(p, text, re.IGNORECASE | re.MULTILINE):
                hit = (m.group(1).strip() if m.lastindex else m.group(0).strip())[:120]
                hit = re.sub(r"^(product|field)\s*\|\s*", "", hit, flags=re.I).strip()
                norm = re.sub(r"\s+", " ", hit.lower())
                if hit and not any(norm == re.sub(r"\s+", " ", s.lower()) for s in seen):
                    seen.append(hit)
                    if first_src is None:
                        first_src = Source(c.file, c.loc, hit[:160])
                if len(seen) >= 4:
                    break
    if not seen or first_src is None:
        return None
    return ("; ".join(seen), first_src)


def offline_extract(chunks: list[Chunk]) -> dict[str, ProductField]:
    state = empty_state()
    NEG = re.compile(r"\b(missing|tbd|tbc|tbd\.|to be confirmed|not found|not declared|not clearly legible|n/?a\b)", re.I)
    for key, pats in _PATTERNS.items():
        if key in _MULTI_KEYS:
            got = _collect_multi(key, pats, chunks)
            if got:
                val, src = got
                state[key] = ProductField(key, val[:400], "found", [src], "medium")
            continue
        for c in chunks:
            text = _clean_text(c.text)
            hit = None
            for p in pats:
                m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
                if m:
                    hit = (m.group(1).strip() if m.lastindex else m.group(0).strip())[:300]
                    break
            if hit:
                # "allergen declaration missing" is NOT a value — treat as missing
                if NEG.search(hit) and key in ("allergens", "tagline", "claims"):
                    continue
                # clean "Product | Mango Oats Bar" -> "Mango Oats Bar", strip xlsx pipes
                clean = re.sub(r"^(product|field)\s*\|\s*", "", hit, flags=re.I).strip()
                clean = re.sub(r"^(product name|ingredients?|claims?|logo)\s*[:\-|]\s*", "", clean, flags=re.I).strip()
                if key in ("serving_size", "net_quantity"):
                    # xlsx rows join cells with " | " — keep the segment with a number
                    segs = [s.strip(" :-") for s in clean.split("|")]
                    num = next((s for s in segs if re.search(r"\d", s)), "")
                    clean = (num or segs[0] if segs else clean).strip()
                    clean = re.sub(r"^(serving size)\s*", "", clean, flags=re.I).strip()
                src = Source(c.file, c.loc, hit[:160])
                # "TBC / TBD" means found-but-needs-review, never drop branding rule
                if NEG.search(hit):
                    state[key] = ProductField(key, clean, "needs_review", [src], "low")
                else:
                    state[key] = ProductField(key, clean, "found", [src], "medium")
                break
    # Normalize region to US-FDA / IN-FSSAI
    _r = state.get("region")
    if _r and _r.value:
        low = _r.value.lower()
        if any(k in low for k in ("united states", "u.s", "north america")):
            _r.value = "US-FDA (United States)"
            _r.status = "found"
        elif "india" in low:
            _r.value = "IN-FSSAI (India)"
            _r.status = "found"
    # tagline/design notes rarely in spec — leave missing intentionally
    return state


def _llm_config() -> tuple[str, str, str, int] | None:
    """Resolve (base_url, api_key, model, timeout) or None when no key is set.

    Precedence: explicit LLM_* vars > GROQ_* alias > NVIDIA_* alias.
    Each alias carries its provider's endpoint + model defaults, so setting
    just GROQ_API_KEY alone is enough for a working Groq backend.
    """
    if os.getenv("LLM_API_KEY"):
        key = os.getenv("LLM_API_KEY", "")
        base = (os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        model = os.getenv("LLM_MODEL") or "llama-3.3-70b-versatile"
    elif os.getenv("GROQ_API_KEY"):
        key = os.getenv("GROQ_API_KEY", "")
        base = "https://api.groq.com/openai/v1"
        model = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
    elif os.getenv("NVIDIA_API_KEY"):
        key = os.getenv("NVIDIA_API_KEY", "")
        base = (os.getenv("NVIDIA_BASE_URL")
                or "https://integrate.api.nvidia.com/v1").rstrip("/")
        model = (os.getenv("NVIDIA_MODEL")
                 or "nvidia/nemotron-3-ultra-550b-a55b")
    else:
        return None
    timeout = int(os.getenv("LLM_TIMEOUT") or os.getenv("NVIDIA_TIMEOUT") or "120")
    return base, key, model, timeout


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces so JSON parsing survives."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # fallback: unclosed trace — keep whatever follows the last </think>
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _llm_extract(prompt: str) -> dict:
    """POST to any OpenAI-compatible /chat/completions endpoint (stdlib only).

    Reasoning models get thinking disabled: our task is a single JSON
    extraction, and the trace only adds latency + parse risk. Providers that
    don't understand the extra_body hint ignore it harmlessly.
    """
    import urllib.request
    cfg = _llm_config()
    assert cfg is not None, "no LLM key configured"
    base, key, model, timeout = cfg
    body = json.dumps({
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt[:12000]}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    msg = payload["choices"][0]["message"]
    # Reasoning models (Nemotron, gpt-oss, Glimmer) put output in
    # reasoning_content/reasoning with content=null — prefer real content first.
    text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    return json.loads(_strip_thinking(str(text)))


# Back-compat alias (old name used in earlier revisions).
_nvidia_available = lambda: _llm_config() is not None  # noqa: E731


# Last LLM failure, human-readable + key-free (shown in UI when falling back).
last_error: str = ""


def _sanitize_error(e: Exception) -> str:
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", errors="ignore")[:200]
        except Exception:
            body = ""
        return f"HTTP {e.code} {body}".strip()
    return f"{type(e).__name__}: {str(e)[:120]}".strip()


def extract_with_llm(chunks: list[Chunk]) -> tuple[dict[str, ProductField], str]:
    """Any OpenAI-compatible LLM backend with agentic multi-agent pipeline.

    When an LLM key is configured, runs the advanced agentic pipeline:
    Classifier → Extractor → Verifier → Conflict Resolver → Router,
    with a reflexive self-correction loop (HEAR pattern).

    Falls back to deterministic offline rules when no key is configured
    or when the LLM call fails.
    """
    global last_error
    last_error = ""
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "extract.txt")
    try:
        template = open(prompt_path, encoding="utf-8").read()
    except Exception:
        template = ""
    combined = _combined(chunks)

    if _llm_config() is not None:
        try:
            # Advanced AI-native path: multi-agent pipeline with LLM
            result = run_agentic_pipeline(chunks)
            _, _, model, _ = _llm_config()  # type: ignore[misc]
            backend = f"agentic-llm-{model.split('/')[-1]}"
            return result["extracted"], backend
        except Exception as e:
            # Never leak key material: scrub anything resembling a secret.
            last_error = re.sub(r"(sk-|gsk_|nvapi-|AIza)[A-Za-z0-9_.-]+",
                                r"\1…", _sanitize_error(e))

    # Deterministic fallback (unchanged behavior)
    return offline_extract(chunks), "offline-rules"


def _from_llm_json(data: dict, chunks: list[Chunk]) -> dict[str, ProductField]:
    state = empty_state()
    fields = data.get("fields", data) if isinstance(data, dict) else {}
    for key, pf in state.items():
        item = fields.get(key)
        if isinstance(item, dict) and item.get("value"):
            quote = str(item.get("quote", item.get("value", "")))[:160]
            src = _find_quote(chunks, quote) or _find_quote(chunks, str(item.get("value", ""))[:60])
            sources = [src] if src else []
            conf = item.get("confidence", "medium")
            if conf not in ("high", "medium", "low"):
                conf = "medium"
            # Anti-hallucination: no verbatim source -> downgrade to needs_review
            status = "found" if sources else "needs_review"
            state[key] = ProductField(key, str(item["value"])[:500], status, sources, conf)
    return state
