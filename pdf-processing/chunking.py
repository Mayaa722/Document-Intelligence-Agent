"""Page-bounded recursive chunking for the Document Intelligence Agent.

Reads the page-level JSON produced by ``pdf_processing.process_pdf`` and returns
chunk-level JSON. Method (see the approved comparison document):

    * Each PDF page is chunked on its own, so a chunk never contains text from two pages.
    * Recursive splitting: paragraphs first, then lines, then sentences, then words,
      and finally a hard cut for a single "word" that is longer than the chunk size.
    * Neighbouring chunks on the same page can overlap by a configurable amount.
    * Every chunk keeps ``doc_id``, ``pdf_page_index``, ``printed_page_label`` and
      ``chunk_index``, plus character offsets into the page's ``text``.

No embedding model or tokenizer is assumed. Sizes are measured in characters by
default. When the tokenizer is chosen, pass its length function to ``chunk_document``
(for example ``length_function=lambda s: len(tokenizer.encode(s))``) and set
``chunk_size`` and ``chunk_overlap`` in tokens. Nothing else needs to change.

The default size and overlap are provisional starting values, not tuned results.
This module uses only the standard library and does not import ``pdf_processing``;
it depends only on the page contract (``doc_id``, ``pdf_page_index``,
``printed_page_label``, ``text``, ``source``, ``quality``). Input pages are never modified.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

__version__ = "0.1.0"

LengthFn = Callable[[str], int]
Span = Tuple[int, int]          # (start, end) character offsets, end exclusive


@dataclass
class ChunkingConfig:
    """All chunking settings in one place so runs can be logged and reproduced."""
    chunk_size: int = 1000        # provisional: maximum chunk length, in ``length_unit``
    chunk_overlap: int = 150      # provisional: target overlap between neighbouring chunks
    length_unit: str = "characters"   # label only; set it when you pass a custom length_function

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def to_dict(self) -> dict:
        return asdict(self)


# Splitting levels, coarse to fine. Each pattern marks the END of a piece, so the
# separator stays attached to the piece before it and no characters are lost.
_LEVELS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("paragraph", re.compile(r"\n\s*\n")),
    ("line", re.compile(r"\n")),
    ("sentence", re.compile(r"[.!?][\"')\]]*\s+")),
    ("word", re.compile(r"\s+")),
)


def _cut(text: str, start: int, end: int, pattern: "re.Pattern[str]") -> List[Span]:
    """Cut text[start:end] after every separator match. Pieces cover the span exactly."""
    pieces: List[Span] = []
    pos = start
    for m in pattern.finditer(text, start, end):
        if m.end() > pos:
            pieces.append((pos, m.end()))
            pos = m.end()
    if pos < end:
        pieces.append((pos, end))
    # A whitespace-only piece (e.g. a lone newline) carries no text: attach it to its neighbour.
    merged: List[Span] = []
    for p in pieces:
        if merged and not text[p[0]:p[1]].strip():
            merged[-1] = (merged[-1][0], p[1])
        else:
            merged.append(p)
    if len(merged) > 1 and not text[merged[0][0]:merged[0][1]].strip():
        merged[1] = (merged[0][0], merged[1][1])
        del merged[0]
    return merged


def _fit(text: str, start: int, end: int, length: LengthFn) -> int:
    """Length of text[start:end] as it would appear in a chunk (surrounding whitespace trimmed)."""
    return length(text[start:end].strip())


def _hard_split(text: str, start: int, end: int, size: int, length: LengthFn) -> List[Span]:
    """Last resort for an unbroken run longer than ``size``: cut it by characters."""
    pieces: List[Span] = []
    pos = start
    while pos < end:
        lo, hi = pos + 1, end            # largest cut in [pos+1, end] that still fits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _fit(text, pos, mid, length) <= size:
                lo = mid
            else:
                hi = mid - 1
        pieces.append((pos, lo))
        pos = lo
    return pieces


def _collect(text: str, start: int, end: int, level: int, size: int, length: LengthFn,
             leaves: List[Span], barriers: set) -> None:
    """Split text[start:end] coarse to fine into ``leaves``: contiguous pieces that each fit.

    ``barriers`` holds the indices of leaves that must start a new chunk. A piece that is
    too big for ``size`` is split on its own, with a barrier before and after it, so a chunk
    is never made of the end of one paragraph (or line, or sentence) glued to the start of
    the next one when that paragraph had to be split.
    """
    if _fit(text, start, end, length) <= size:
        leaves.append((start, end))
        return
    if level >= len(_LEVELS):
        leaves.extend(_hard_split(text, start, end, size, length))
        return
    pieces = _cut(text, start, end, _LEVELS[level][1])
    if len(pieces) <= 1:                 # this separator does not occur: try the next level
        _collect(text, start, end, level + 1, size, length, leaves, barriers)
        return
    for ps, pe in pieces:
        if _fit(text, ps, pe, length) <= size:
            leaves.append((ps, pe))
        else:
            barriers.add(len(leaves))
            _collect(text, ps, pe, level + 1, size, length, leaves, barriers)
            barriers.add(len(leaves))


def _pack(text: str, leaves: List[Span], barriers: set, cfg: ChunkingConfig,
          length: LengthFn) -> List[Span]:
    """Greedily pack leaves into chunks of at most ``chunk_size``, with overlap.

    A chunk starts with trailing leaves of the previous chunk when their total length is at
    most ``chunk_overlap`` and the new chunk still fits. Overlap is made of whole leaves, so
    it can be smaller than ``chunk_overlap``, or zero when the last leaf of a chunk is longer
    than that. Chunks never start in the middle of a leaf.
    """
    spans: List[Span] = []
    n = len(leaves)
    i = 0            # first leaf that belongs to the chunk being built (not overlap)
    ovl = 0          # first leaf of the chunk, including any overlap leaves before ``i``
    while i < n:
        j = i
        while (j + 1 < n and (j + 1) not in barriers
               and _fit(text, leaves[ovl][0], leaves[j + 1][1], length) <= cfg.chunk_size):
            j += 1
        spans.append((leaves[ovl][0], leaves[j][1]))
        if j == n - 1:
            break
        nxt = j + 1                       # default: no overlap
        if cfg.chunk_overlap > 0:
            for c in range(i + 1, j + 1):          # smallest c = largest overlap that fits
                if (_fit(text, leaves[c][0], leaves[j][1], length) <= cfg.chunk_overlap
                        and _fit(text, leaves[c][0], leaves[j + 1][1], length) <= cfg.chunk_size):
                    nxt = c
                    break
        i, ovl = j + 1, nxt               # i grows every round, so the loop always ends
    return spans


def split_text(text: str, config: Optional[ChunkingConfig] = None,
               length_function: Optional[LengthFn] = None) -> List[Span]:
    """Split one page's text into chunk spans ``(char_start, char_end)``.

    ``text[char_start:char_end]`` is the chunk text, trimmed of surrounding whitespace.
    Whitespace-only chunks are dropped.
    """
    cfg = config or ChunkingConfig()
    cfg.validate()
    length = length_function or len
    if not text or not text.strip():
        return []
    leaves: List[Span] = []
    barriers: set = set()
    _collect(text, 0, len(text), 0, cfg.chunk_size, length, leaves, barriers)
    result: List[Span] = []
    for s, e in _pack(text, leaves, barriers, cfg, length):
        chunk = text[s:e]
        lead = len(chunk) - len(chunk.lstrip())
        s2, e2 = s + lead, e - (len(chunk) - len(chunk.rstrip()))
        if e2 > s2:
            result.append((s2, e2))
    return result


def chunk_page(page: dict, config: Optional[ChunkingConfig] = None,
               length_function: Optional[LengthFn] = None,
               doc_id: Optional[str] = None) -> List[dict]:
    """Chunk one page dict from ``process_pdf``. Chunks carry no document-wide index yet."""
    cfg = config or ChunkingConfig()
    length = length_function or len
    text = page.get("text") or ""
    doc_id = page.get("doc_id") or doc_id
    quality = page.get("quality") or {}
    source_type = (page.get("source") or {}).get("type")
    chunks = []
    for page_chunk_index, (s, e) in enumerate(split_text(text, cfg, length)):
        piece = text[s:e]
        chunks.append({
            "doc_id": doc_id,
            "pdf_page_index": page["pdf_page_index"],
            "printed_page_label": page.get("printed_page_label"),
            "page_chunk_index": page_chunk_index,
            "text": piece,
            "char_start": s,
            "char_end": e,
            "length": length(piece),
            "source_type": source_type,
            "flags": list(quality.get("flags", [])),
        })
    return chunks


def chunk_pages(pages: List[dict], config: Optional[ChunkingConfig] = None,
                length_function: Optional[LengthFn] = None,
                doc_id: Optional[str] = None) -> Tuple[List[dict], List[str]]:
    """Chunk pages in the given order. Returns ``(chunks, warnings)``.

    ``chunk_index`` runs 0, 1, 2, ... over the whole document; ``page_chunk_index``
    restarts at 0 on every page. ``pdf_page_index`` stays 1-based, as in pdf_processing.
    """
    cfg = config or ChunkingConfig()
    chunks: List[dict] = []
    warnings: List[str] = []
    for page in pages:
        page_chunks = chunk_page(page, cfg, length_function, doc_id)
        if not page_chunks:
            warnings.append(f"Page {page['pdf_page_index']} has no text; no chunks created.")
        for ch in page_chunks:
            ch["chunk_index"] = len(chunks)
            ch["chunk_id"] = f"{ch['doc_id']}_p{ch['pdf_page_index']}_c{ch['chunk_index']}"
            chunks.append(ch)
    return chunks, warnings


def chunk_document(result: dict, config: Optional[ChunkingConfig] = None,
                   length_function: Optional[LengthFn] = None) -> dict:
    """Chunk a ``process_pdf`` result (or one loaded with ``load_result_json``).

    Returns ``{doc_id, source_file, page_count, config, chunks, warnings}``.
    """
    cfg = config or ChunkingConfig()
    chunks, warnings = chunk_pages(result["pages"], cfg, length_function, result.get("doc_id"))
    return {"doc_id": result.get("doc_id"), "source_file": result.get("source_file"),
            "page_count": len(result["pages"]), "config": cfg.to_dict(),
            "chunks": chunks, "warnings": warnings}


# JSON output

def save_chunks_json(chunked: dict, path: str) -> None:
    """Write a chunk_document() result to a UTF-8 JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chunked, fh, ensure_ascii=False, indent=2)


def load_chunks_json(path: str) -> dict:
    """Load a result written by save_chunks_json()."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
