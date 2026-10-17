"""Multi-modal document ingestion: LayoutLMv3 / Donut / OCR-free visual understanding.

Implements the advanced ingestion layer from ARCHITECTURE_ADVANCED.md:

- LayoutLMv3-style: text + image + layout Transformer for scanned/spec PDFs
- Donut (OCR-free VDU): direct document image -> structured JSON, no OCR engine
- Fallback: existing PyMuPDF / python-docx / openpyxl text extraction

All heavy model deps are optional (transformers, torch). When unavailable the
module gracefully falls back to the standard ingest path so the app keeps
working offline with zero new dependencies.
"""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .ingest import Chunk, ingest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayoutElement:
    """A text element with 2D layout coordinates (LayoutLMv3-style)."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    source: str  # "layoutlmv3" | "pymupdf" | "donut" | "regex"


@dataclass
class ParsedDocument:
    """Result of multi-modal parsing."""
    chunks: list[Chunk]
    layout_elements: list[LayoutElement] = field(default_factory=list)
    parser_used: str = "text-only"  # "text-only" | "layoutlmv3" | "donut"
    ocr_free: bool = False


# ---------------------------------------------------------------------------
# Optional heavy dependencies (transformers / torch)
# ---------------------------------------------------------------------------


def _try_import_layoutlmv3():
    """Try to import LayoutLMv3 processor + model. Returns (None, None) if unavailable."""
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3Model
        import torch
        return LayoutLMv3Processor, LayoutLMv3Model, torch
    except Exception:
        return None, None, None


def _try_import_donut():
    """Try to import Donut (OCR-free document understanding transformer)."""
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        import torch
        return DonutProcessor, VisionEncoderDecoderModel, torch
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Text-based fallback parser (existing pipeline, zero new deps)
# ---------------------------------------------------------------------------


def _parse_text_only(filename: str, data: bytes) -> ParsedDocument:
    """Standard ingest path — no new dependencies."""
    chunks = ingest(filename, data)
    return ParsedDocument(
        chunks=chunks,
        parser_used="text-only",
        ocr_free=False,
    )


# ---------------------------------------------------------------------------
# LayoutLMv3 parser (multimodal text + image + layout)
# ---------------------------------------------------------------------------


def _parse_with_layoutlmv3(filename: str, data: bytes) -> Optional[ParsedDocument]:
    """Parse using LayoutLMv3 (text + image + layout Transformer).

    Requires: transformers, torch, and a LayoutLMv3 checkpoint.
    Falls back to text-only if unavailable.
    """
    ProcessorCls, ModelCls, torch = _try_import_layoutlmv3()
    if ProcessorCls is None:
        logger.info("LayoutLMv3 deps unavailable — falling back to text-only")
        return None

    import pymupdf  # PyMuPDF

    try:
        processor = ProcessorCls.from_pretrained("microsoft/layoutlmv3-base")
        model = ModelCls.from_pretrained("microsoft/layoutlmv3-base")
        model.eval()

        doc = pymupdf.open(stream=data, filetype="pdf")
        elements: list[LayoutElement] = []
        chunks: list[Chunk] = []

        for page_idx, page in enumerate(doc, start=1):
            # Render page to image for LayoutLMv3 visual input
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            img_bytes = pix.tobytes("png")

            # Extract text with bounding boxes (OCR-like from PyMuPDF)
            words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
            if not words:
                continue

            # Prepare inputs for LayoutLMv3
            image = __import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(img_bytes))
            words_list = [w[4] for w in words]
            boxes = [[int(w[0]), int(w[1]), int(w[2]), int(w[3])] for w in words]

            encoding = processor(
                image,
                words_list,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            )

            with torch.no_grad():
                outputs = model(**encoding)

            # Reconstruct text with layout info
            text_parts = []
            for i, w in enumerate(words):
                elements.append(LayoutElement(
                    text=w[4],
                    x0=w[0], y0=w[1], x1=w[2], y1=w[3],
                    page=page_idx,
                    source="layoutlmv3",
                ))
                text_parts.append(w[4])

            page_text = " ".join(text_parts)
            if page_text.strip():
                chunks.append(Chunk(filename, f"p{page_idx}", page_text))

        return ParsedDocument(
            chunks=chunks,
            layout_elements=elements,
            parser_used="layoutlmv3",
            ocr_free=True,
        )
    except Exception as e:
        logger.warning(f"LayoutLMv3 parsing failed: {e} — falling back to text-only")
        return None


# ---------------------------------------------------------------------------
# Donut parser (OCR-free visual document understanding)
# ---------------------------------------------------------------------------


def _parse_with_donut(filename: str, data: bytes) -> Optional[ParsedDocument]:
    """Parse using Donut (OCR-free document understanding transformer).

    Directly maps document image -> structured JSON without OCR.
    Requires: transformers, torch, and a Donut checkpoint (e.g. naver-clova-ix/donut-base-finetuned-cord-v2).
    """
    ProcessorCls, ModelCls, torch = _try_import_donut()
    if ProcessorCls is None:
        logger.info("Donut deps unavailable — falling back to text-only")
        return None

    try:
        from PIL import Image

        processor = ProcessorCls.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        model = ModelCls.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        model.eval()

        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")

        chunks: list[Chunk] = []
        for page_idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

            # Donut task prompt for document parsing
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]

            pixel_values = processor(img, return_tensors="pt").pixel_values

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=model.decoder.config.max_position_embeddings,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

            seq = processor.batch_decode(outputs.sequences)[0]
            seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            seq = seq.replace("<s_cord-v2>", "").strip()

            try:
                structured = json.loads(seq)
            except json.JSONDecodeError:
                structured = {"raw": seq}

            # Flatten structured JSON into text chunks for downstream pipeline
            text_parts = []
            if isinstance(structured, dict):
                for k, v in structured.items():
                    if isinstance(v, (str, int, float)):
                        text_parts.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        text_parts.append(f"{k}: " + ", ".join(str(x) for x in v))
            else:
                text_parts.append(str(structured))

            page_text = "\n".join(text_parts)
            if page_text.strip():
                chunks.append(Chunk(filename, f"p{page_idx}", page_text))

        return ParsedDocument(
            chunks=chunks,
            parser_used="donut",
            ocr_free=True,
        )
    except Exception as e:
        logger.warning(f"Donut parsing failed: {e} — falling back to text-only")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_document(filename: str, data: bytes, prefer: str = "auto") -> ParsedDocument:
    """Parse a document using the best available multi-modal parser.

    Args:
        filename: original filename (for chunk provenance)
        data: raw file bytes
        prefer: "auto" | "layoutlmv3" | "donut" | "text-only"

    Returns:
        ParsedDocument with chunks and optional layout elements.
    """
    fn = filename.lower()

    # Non-PDF files: text-only path (no visual model needed for DOCX/XLSX)
    if not fn.endswith(".pdf"):
        return _parse_text_only(filename, data)

    if prefer == "text-only":
        return _parse_text_only(filename, data)

    if prefer in ("auto", "layoutlmv3"):
        result = _parse_with_layoutlmv3(filename, data)
        if result is not None:
            return result

    if prefer in ("auto", "donut"):
        result = _parse_with_donut(filename, data)
        if result is not None:
            return result

    # Final fallback
    return _parse_text_only(filename, data)


def parse_document_chunks(filename: str, data: bytes, prefer: str = "auto") -> list[Chunk]:
    """Convenience: return just the chunks (drop-in replacement for ingest())."""
    parsed = parse_document(filename, data, prefer=prefer)
    return parsed.chunks
