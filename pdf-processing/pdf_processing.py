"""PDF processing: validation, text extraction, OCR and English cleaning.

Methods (see the approved comparison document):
    * Validation : signature check, open test, size/page limits, structured error codes
    * Extraction : pypdfium2 (per page, text layer)
    * OCR        : Tesseract (English) with light image preprocessing, only for routed pages
    * Cleaning   : conservative English rules; the source texts are never modified

Output is one JSON-serialisable dict per page (the page-level JSON contract) that the
chunker reads. Chunking and evaluation questions are not part of this module.

``raw_text`` always holds the original pypdfium2 output. When a page is OCR'd, the OCR
result is stored separately in ``ocr_text`` and ``source`` says which text was cleaned.
If OCR returns nothing but the text layer has text, the text layer is used.

Thresholds marked "provisional" are starting values to be tuned in experiments E3/E4.
Everything is CPU-only and uses free, open-source tools.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image

__version__ = "0.1.0"

# Error codes and configuration

FILE_NOT_FOUND = "FILE_NOT_FOUND"
EMPTY_FILE = "EMPTY_FILE"
NOT_A_PDF = "NOT_A_PDF"
TOO_LARGE = "TOO_LARGE"
CORRUPT = "CORRUPT"
ENCRYPTED = "ENCRYPTED"
ZERO_PAGES = "ZERO_PAGES"
TOO_MANY_PAGES = "TOO_MANY_PAGES"

ERROR_CODES = (FILE_NOT_FOUND, EMPTY_FILE, NOT_A_PDF, TOO_LARGE, CORRUPT,
               ENCRYPTED, ZERO_PAGES, TOO_MANY_PAGES)


@dataclass
class CleaningConfig:
    """Switches for the conservative English cleaning rules (each can be turned off)."""
    normalize_unicode: bool = True        # NFC, ligature expansion, control/zero-width chars
    collapse_whitespace: bool = True      # runs of spaces/tabs, trailing spaces, blank-line runs
    dehyphenate: bool = True              # "informa-\ntion" -> "information"
    rejoin_lines: bool = True             # join wrapped prose lines (never short/table-like lines)
    remove_header_footer: bool = False    # off by default: can drop real content on short pages; drops lines repeated across pages
    extract_page_numbers: bool = True     # standalone page numbers -> printed_page_label
    make_search_text: bool = True         # lowercase, single-space copy for lexical search
    min_join_line_len: int = 40           # provisional: only join lines at least this long
    header_footer_zone_lines: int = 2     # provisional: look at first/last N lines of a page
    header_footer_min_fraction: float = 0.5   # provisional: share of pages a line must repeat on
    header_footer_min_pages: int = 4      # provisional: skip the rule on very short documents


@dataclass
class PipelineConfig:
    """All limits and thresholds in one place so runs can be logged and reproduced."""
    # validation limits
    max_file_mb: float = 100.0
    max_pages: int = 1000
    # routing (provisional; tuned in E3)
    min_chars_text_layer: int = 25
    image_coverage_for_ocr: float = 0.5
    garbled_ratio_for_ocr: float = 0.2
    # OCR
    ocr_enabled: bool = True
    ocr_lang: str = "eng"
    ocr_dpi: int = 300
    ocr_psm: int = 3
    ocr_deskew: bool = True
    ocr_binarize: bool = True
    low_ocr_confidence: float = 60.0      # provisional: mean word confidence below this -> flag
    # cleaning
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)

    def to_dict(self) -> dict:
        return asdict(self)


class PDFValidationError(Exception):
    """Raised when a file is rejected; ``code`` is one of ERROR_CODES."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class ValidationResult:
    ok: bool
    error_code: Optional[str] = None
    message: str = ""
    page_count: int = 0
    file_size_mb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Validation

def _fail(code: str, message: str, **kw) -> ValidationResult:
    return ValidationResult(ok=False, error_code=code, message=message, **kw)


def validate_pdf(path: str, config: Optional[PipelineConfig] = None) -> ValidationResult:
    """Check a file before processing. Never raises; inspect ``result.ok`` / ``error_code``.

    Checks, in order: exists, non-empty, size limit, ``%PDF-`` signature in the first
    1024 bytes, opens with pypdfium2 (password-protected -> ENCRYPTED, otherwise CORRUPT),
    at least one page, page-count limit.
    Note: PDFs encrypted with an empty user password open normally and are accepted.
    """
    config = config or PipelineConfig()
    if not path or not os.path.isfile(path):
        return _fail(FILE_NOT_FOUND, f"File not found: {path}")
    size = os.path.getsize(path)
    size_mb = size / (1024 * 1024)
    if size == 0:
        return _fail(EMPTY_FILE, "File is empty (0 bytes).")
    if size_mb > config.max_file_mb:
        return _fail(TOO_LARGE, f"File is {size_mb:.1f} MB; limit is {config.max_file_mb} MB.",
                     file_size_mb=size_mb)
    with open(path, "rb") as fh:
        head = fh.read(1024)
    if b"%PDF-" not in head:
        return _fail(NOT_A_PDF, "Missing %PDF- signature; not a PDF file.", file_size_mb=size_mb)

    try:
        pdf = pdfium.PdfDocument(path)
    except Exception as exc:  # pypdfium2 raises PdfiumError; keep broad so nothing leaks out
        text = str(exc).lower()
        if "password" in text:
            return _fail(ENCRYPTED, "PDF is password-protected.", file_size_mb=size_mb)
        return _fail(CORRUPT, f"PDF could not be opened: {exc}", file_size_mb=size_mb)
    try:
        n = len(pdf)
    finally:
        pdf.close()
    if n == 0:
        return _fail(ZERO_PAGES, "PDF has no pages.", file_size_mb=size_mb)
    if n > config.max_pages:
        return _fail(TOO_MANY_PAGES, f"PDF has {n} pages; limit is {config.max_pages}.",
                     page_count=n, file_size_mb=size_mb)
    return ValidationResult(ok=True, message="OK", page_count=n, file_size_mb=size_mb)


# Text extraction and per-page flags

def pypdfium2_version() -> str:
    """Installed pypdfium2 version, or "unknown" if it cannot be read."""
    try:
        return str(pdfium.PYPDFIUM_INFO.version)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("pypdfium2")
    except Exception:
        return "unknown"


_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
              "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
_CID_RE = re.compile(r"\(cid:\d+\)")


def garbled_ratio(text: str) -> float:
    """Share of non-space characters that look like extraction garbage.

    Counts U+FFFD, private-use characters, control characters (other than newline/tab)
    and ``(cid:123)`` placeholders.
    """
    if not text:
        return 0.0
    bad_cid = sum(len(m.group(0)) for m in _CID_RE.finditer(text))
    bad = bad_cid
    total = 0
    for ch in _CID_RE.sub("", text):
        if ch.isspace() or ch == "\x02":     # \x02 is pdfium's marker for a line-end hyphen
            continue
        total += 1
        cat = unicodedata.category(ch)
        if ch == "\ufffd" or cat in ("Co", "Cc", "Cs"):
            bad += 1
    total += bad_cid
    return bad / total if total else 0.0


def _image_coverage(page) -> Optional[float]:
    """Fraction of the page area covered by image objects (capped at 1.0).

    Returns None when it cannot be measured, so a failure is never mistaken for
    "no images".
    """
    try:
        width, height = page.get_size()
        area = float(width * height)
        if area <= 0:
            return None
        covered = 0.0
        for obj in page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]):
            getter = getattr(obj, "get_bounds", None) or obj.get_pos
            left, bottom, right, top = getter()
            covered += max(0.0, right - left) * max(0.0, top - bottom)
    except Exception:
        return None
    return min(1.0, covered / area)


def extract_page(page) -> dict:
    """Extract one pypdfium2 page: raw text plus the flags used for routing.

    ``raw_text`` is exactly what pypdfium2 returned (no cleaning). ``image_coverage``
    is None if it could not be measured.
    """
    textpage = page.get_textpage()
    try:
        raw_text = textpage.get_text_bounded() or ""
    finally:
        textpage.close()
    coverage = _image_coverage(page)
    return {
        "raw_text": raw_text,
        "char_count": len(raw_text.strip()),
        "image_coverage": None if coverage is None else round(coverage, 3),
        "garbled_ratio": round(garbled_ratio(raw_text), 3),
    }


def decide_route(flags: dict, config: Optional[PipelineConfig] = None) -> Tuple[str, str]:
    """Per-page routing: return ("text_layer" | "ocr" | "empty", reason).

    * garbled text layer                      -> ocr
    * (almost) no text but page is image-like -> ocr
    * (almost) no text, image check failed    -> ocr (safer than assuming the page is blank)
    * (almost) no text and no image           -> empty (blank page, no OCR)
    * otherwise                               -> text_layer
    """
    config = config or PipelineConfig()
    if flags["garbled_ratio"] >= config.garbled_ratio_for_ocr and flags["char_count"] > 0:
        return "ocr", "garbled_text_layer"
    if flags["char_count"] < config.min_chars_text_layer:
        coverage = flags["image_coverage"]
        if coverage is None:
            return "ocr", "no_text_layer_image_check_failed"
        if coverage >= config.image_coverage_for_ocr:
            return "ocr", "no_text_layer_image_page"
        return "empty", "no_text_and_no_image"
    return "text_layer", "valid_text_layer"


# OCR

def tesseract_version() -> Optional[str]:
    """Return the Tesseract version string, or None if it is not installed."""
    try:
        import pytesseract
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None


def render_page(page, dpi: int = 300) -> Image.Image:
    """Render a pypdfium2 page to a PIL RGB image at the given DPI."""
    return page.render(scale=dpi / 72.0).to_pil().convert("RGB")


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = 0.0
    sum_b = 0.0
    best, best_t = -1.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, best_t = between, t
    return best_t


def estimate_skew_angle(img: Image.Image, max_angle: float = 5.0, step: float = 0.5) -> float:
    """Estimate page skew in degrees (projection-profile search on a small copy).

    Returns the angle in degrees (PIL ``Image.rotate`` convention, counter-clockwise
    positive) by which the image should be rotated to straighten it.
    """
    gray = img.convert("L")
    gray.thumbnail((700, 700))
    arr = np.array(gray)
    if arr.std() < 1.0:
        return 0.0
    ink = (arr < _otsu_threshold(arr)).astype(np.uint8) * 255
    if ink.sum() == 0:
        return 0.0
    ink_img = Image.fromarray(ink)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        rotated = np.array(ink_img.rotate(float(angle), resample=Image.BILINEAR, fillcolor=0))
        profile = rotated.sum(axis=1).astype(np.float64)
        score = float(np.var(profile))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def preprocess_image(img: Image.Image, deskew: bool = True, binarize: bool = True) -> Image.Image:
    """Light preprocessing for OCR: grayscale, auto-contrast, optional deskew and Otsu binarisation."""
    from PIL import ImageOps
    gray = ImageOps.autocontrast(img.convert("L"))
    if deskew:
        angle = estimate_skew_angle(gray)
        if abs(angle) >= 0.25:
            gray = gray.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    if binarize:
        arr = np.array(gray)
        thr = _otsu_threshold(arr)
        gray = Image.fromarray(((arr > thr) * 255).astype(np.uint8))
    return gray


def ocr_image(img: Image.Image, config: Optional[PipelineConfig] = None) -> dict:
    """Run Tesseract on a PIL image. Returns text, mean word confidence and word count.

    ``word_count`` counts every recognized word; ``mean_confidence`` only uses words that
    have a valid (non-negative) confidence score.

    Raises RuntimeError if pytesseract / the Tesseract binary is unavailable.
    """
    config = config or PipelineConfig()
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise RuntimeError("pytesseract is not installed.") from exc
    if tesseract_version() is None:
        raise RuntimeError("Tesseract binary not found. Install tesseract-ocr.")

    data = pytesseract.image_to_data(
        img, lang=config.ocr_lang, config=f"--oem 1 --psm {config.ocr_psm}",
        output_type=Output.DICT)
    lines: Dict[Tuple[int, int, int], List[str]] = {}
    order: List[Tuple[int, int, int]] = []
    confs: List[float] = []
    n_words = 0
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(word)
        n_words += 1
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            confs.append(conf)
    out: List[str] = []
    prev = None
    for key in order:
        if prev is not None and key[:2] != prev[:2]:
            out.append("")            # blank line between paragraphs/blocks
        out.append(" ".join(lines[key]))
        prev = key
    return {"text": "\n".join(out), "mean_confidence": round(float(np.mean(confs)), 1) if confs else None,
            "word_count": n_words}


def ocr_page(page, config: Optional[PipelineConfig] = None) -> dict:
    """Render a pypdfium2 page, preprocess it and OCR it."""
    config = config or PipelineConfig()
    img = render_page(page, config.ocr_dpi)
    img = preprocess_image(img, deskew=config.ocr_deskew, binarize=config.ocr_binarize)
    return ocr_image(img, config)


# English cleaning

_PAGE_NUM_RE = re.compile(
    r"^\s*(?:page\s+(\d{1,4})|(\d{1,3}))(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)
_DEHYPHEN_RE = re.compile(r"([A-Za-z]{2,})-[ \t]*\n[ \t]*(?=[a-z])")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def normalize_unicode_text(text: str) -> str:
    """NFC, ligature expansion, newline normalisation, drop control/zero-width/soft-hyphen chars."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = text.replace("\u00ad", "")                 # soft hyphen
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")


def collapse_whitespace_text(text: str) -> str:
    """Collapse runs of spaces/tabs, strip line ends, limit blank lines to one."""
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def dehyphenate_text(text: str) -> str:
    """Join words split by a line-end hyphen when the next fragment starts lowercase."""
    return _DEHYPHEN_RE.sub(r"\1", text)


def rejoin_lines_text(text: str, min_len: int = 40) -> str:
    """Join wrapped prose lines. Only joins when the previous line is long, has no end
    punctuation, and the next line starts lowercase, so headings, bullets, slide lines
    and table rows stay on their own lines."""
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        if out and line and out[-1]:
            prev = out[-1]
            if (len(prev) >= min_len and re.search(r"[a-z,]$", prev)
                    and re.match(r"[a-z]", line) and "\t" not in prev and "  " not in prev):
                out[-1] = prev + " " + line
                continue
        out.append(line)
    return "\n".join(out)


def _norm_for_match(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def find_page_number(text: str, zone: int = 2) -> Optional[str]:
    """Return a standalone page number found in the first/last ``zone`` lines, else None."""
    lines = [l for l in text.split("\n") if l.strip()]
    for line in lines[-zone:] + lines[:zone]:
        m = _PAGE_NUM_RE.match(line)
        if m:
            return m.group(1) or m.group(2)
    return None


def _strip_page_number(text: str, zone: int = 2) -> str:
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip()]
    for i in idx[-zone:] + idx[:zone]:
        if _PAGE_NUM_RE.match(lines[i]):
            del lines[i]
            break
    return "\n".join(lines)


def find_repeated_lines(page_texts: List[str], cfg: CleaningConfig) -> set:
    """Normalised lines that repeat in the header/footer zone of many pages."""
    n_pages = len(page_texts)
    if n_pages < cfg.header_footer_min_pages:
        return set()
    counts: Counter = Counter()
    for text in page_texts:
        lines = [l for l in text.split("\n") if l.strip()]
        zone = lines[:cfg.header_footer_zone_lines] + lines[-cfg.header_footer_zone_lines:]
        counts.update({_norm_for_match(l) for l in zone})
    need = max(2, int(np.ceil(cfg.header_footer_min_fraction * n_pages)))
    return {line for line, c in counts.items()
            if c >= need and line and not _PAGE_NUM_RE.match(line)}


def _remove_lines(text: str, repeated: set, zone: int) -> Tuple[str, int]:
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip()]
    drop = {i for i in idx[:zone] + idx[-zone:] if _norm_for_match(lines[i]) in repeated}
    kept = [l for i, l in enumerate(lines) if i not in drop]
    return "\n".join(kept), len(drop)


def make_search_text(text: str) -> str:
    """Lowercase, single-spaced copy for lexical (BM25) search. No stemming here."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _text_to_clean(page: dict) -> str:
    """OCR text for OCR'd pages, otherwise the pypdfium2 text."""
    if page.get("source", {}).get("type") == "ocr":
        return page.get("ocr_text") or ""
    return page["raw_text"]


def clean_pages(pages: List[dict], cfg: Optional[CleaningConfig] = None) -> List[dict]:
    """Clean a whole document in place and return it.

    For every page dict this reads the page's source text (``ocr_text`` for OCR'd pages,
    otherwise ``raw_text``) and sets ``text``, ``search_text`` and ``printed_page_label``.
    ``raw_text`` and ``ocr_text`` are never modified.
    Header/footer removal is document-level, so pass all pages of one document together.
    """
    cfg = cfg or CleaningConfig()
    working = []
    for p in pages:
        t = _text_to_clean(p)
        if cfg.normalize_unicode:
            t = normalize_unicode_text(t)
        else:
            t = t.replace("\r\n", "\n")
        working.append(t)

    repeated = (find_repeated_lines([collapse_whitespace_text(t) for t in working], cfg)
                if cfg.remove_header_footer else set())

    for p, t in zip(pages, working):
        removed = 0
        if cfg.collapse_whitespace:
            t = collapse_whitespace_text(t)
        label = None
        if cfg.extract_page_numbers:
            label = find_page_number(t, cfg.header_footer_zone_lines)
            if label is not None:
                t = _strip_page_number(t, cfg.header_footer_zone_lines)
        if repeated:
            t, removed = _remove_lines(t, repeated, cfg.header_footer_zone_lines)
        if cfg.dehyphenate:
            t = dehyphenate_text(t)
        if cfg.rejoin_lines:
            t = rejoin_lines_text(t, cfg.min_join_line_len)
        if cfg.collapse_whitespace:
            t = collapse_whitespace_text(t)
        p["text"] = t
        p["printed_page_label"] = label
        p["search_text"] = make_search_text(t) if cfg.make_search_text else None
        p["quality"]["header_footer_lines_removed"] = removed
    return pages


# Pipeline

def guess_language(text: str) -> str:
    """Tiny English-only heuristic: "en", "non_en" (mostly non-Latin letters) or "unknown"."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return "unknown"
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    if latin / len(letters) < 0.5:
        return "non_en"
    words = re.findall(r"[a-z]+", text.lower())
    common = {"the", "and", "of", "to", "in", "is", "that", "for", "with", "as", "are", "on"}
    return "en" if words and sum(w in common for w in words) / len(words) > 0.03 else "unknown"


def non_latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "LATIN" not in unicodedata.name(c, "")) / len(letters)


def process_pdf(path: str, doc_id: Optional[str] = None,
                config: Optional[PipelineConfig] = None,
                progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Process one PDF and return the document result.

    Raises PDFValidationError if the file is rejected. Otherwise returns::

        {"doc_id", "source_file", "page_count", "config", "pages": [page, ...],
         "warnings": [...]}

    Each page follows the page-level JSON contract (see README).
    """
    config = config or PipelineConfig()
    check = validate_pdf(path, config)
    if not check.ok:
        raise PDFValidationError(check.error_code, check.message)

    doc_id = doc_id or os.path.splitext(os.path.basename(path))[0]
    tess = tesseract_version() if config.ocr_enabled else None
    engine_version = pypdfium2_version()
    doc_warnings: List[str] = []
    pages: List[dict] = []
    pdf = pdfium.PdfDocument(path)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                flags = extract_page(page)
                route, reason = decide_route(flags, config)
                source = {"type": "text_layer", "engine": "pypdfium2", "version": engine_version}
                page_flags: List[str] = []
                page_warnings: List[str] = []
                mean_conf = None
                ocr_text = None                    # set only when OCR is actually used
                source_text = flags["raw_text"]    # text the language check looks at

                if flags["image_coverage"] is None:
                    page_flags.append("image_detection_failed")
                    page_warnings.append("Image coverage could not be measured.")

                if route == "empty":
                    page_flags.append("empty")
                elif route == "ocr":
                    if not config.ocr_enabled:
                        page_flags.append("ocr_skipped")
                        page_warnings.append("Page needs OCR but OCR is disabled.")
                    elif tess is None:
                        page_flags.append("ocr_unavailable")
                        page_warnings.append("Page needs OCR but Tesseract or pytesseract is not available.")
                    else:
                        try:
                            res = ocr_page(page, config)
                            ocr_text, mean_conf = res["text"], res["mean_confidence"]
                            if ocr_text.strip():
                                source = {"type": "ocr", "engine": "tesseract", "version": tess}
                                source_text = ocr_text
                                if mean_conf is not None and mean_conf < config.low_ocr_confidence:
                                    page_flags.append("low_ocr_confidence")
                                    page_warnings.append(f"Mean OCR confidence {mean_conf} is below {config.low_ocr_confidence}.")
                            elif flags["raw_text"].strip():
                                # OCR found nothing: keep the pypdfium2 text rather than lose it
                                page_flags.append("ocr_empty_used_text_layer")
                                page_warnings.append("OCR returned no text; using the pypdfium2 text layer.")
                            else:
                                # nothing from either source
                                source = {"type": "ocr", "engine": "tesseract", "version": tess}
                                page_flags.append("empty")
                        except Exception as exc:
                            page_flags.append("ocr_failed")
                            page_warnings.append(f"OCR failed: {exc}")
                if source["type"] == "text_layer" and flags["garbled_ratio"] >= config.garbled_ratio_for_ocr:
                    page_flags.append("garbled")   # garbled layer that could not be OCR'd

                page_dict = {
                    "doc_id": doc_id,
                    "pdf_page_index": i + 1,
                    "printed_page_label": None,
                    "source": source,
                    "raw_text": flags["raw_text"],     # always the original pypdfium2 output
                    "ocr_text": ocr_text,
                    "text": "",
                    "search_text": None,
                    "quality": {
                        "route": route, "route_reason": reason,
                        "language_guess": guess_language(source_text),
                        "char_count": flags["char_count"],
                        "image_coverage": flags["image_coverage"],
                        "garbled_ratio": flags["garbled_ratio"],
                        "mean_ocr_confidence": mean_conf,
                        "flags": page_flags, "warnings": page_warnings,
                    },
                }
                if non_latin_ratio(source_text) > 0.3:
                    page_dict["quality"]["flags"].append("non_english_script")
                    page_dict["quality"]["warnings"].append(
                        "Mostly non-Latin text; version 1 cleaning is English-only.")
                pages.append(page_dict)
            finally:
                page.close()
            if progress:
                progress(i + 1, len(pdf))
    finally:
        pdf.close()

    clean_pages(pages, config.cleaning)
    if any("ocr_unavailable" in p["quality"]["flags"] for p in pages):
        doc_warnings.append("Some pages need OCR but Tesseract or pytesseract is not available; "
                            "their text may be missing or unreliable.")
    return {"doc_id": doc_id, "source_file": os.path.basename(path), "page_count": len(pages),
            "config": config.to_dict(), "pages": pages, "warnings": doc_warnings}


# JSON output

def save_result_json(result: dict, path: str) -> None:
    """Write a process_pdf() result to a UTF-8 JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


def load_result_json(path: str) -> dict:
    """Load a result written by save_result_json()."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
