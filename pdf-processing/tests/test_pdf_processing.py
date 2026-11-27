"""Essential tests for pdf_processing.py.

Run from the repository root with either:
    python -m unittest discover -s tests -v
    pytest -q
Tests that need an optional package (reportlab, pypdf) or the Tesseract binary are
skipped automatically when it is missing.
"""

import difflib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import pdf_processing as pp  # noqa: E402

try:
    from reportlab.pdfgen import canvas as rl_canvas
except ImportError:  # pragma: no cover
    rl_canvas = None
try:
    from pypdf import PdfReader, PdfWriter
except ImportError:  # pragma: no cover
    PdfReader = PdfWriter = None

HAS_TESSERACT = pp.tesseract_version() is not None


def make_text_pdf(path, pages):
    """Digital PDF; ``pages`` is a list of lists of text lines."""
    c = rl_canvas.Canvas(path)
    for lines in pages:
        y = 780
        for line in lines:
            c.drawString(60, y, line)
            y -= 18
        c.showPage()
    c.save()


def make_scanned_pdf(path, lines, angle=0.0):
    """Image-only PDF (no text layer) containing the given lines of large text."""
    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=64)
    y = 200
    for line in lines:
        draw.text((150, y), line, fill="black", font=font)
        y += 140
    if angle:
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor="white")
    img.save(path, "PDF", resolution=200.0)


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def p(self, name):
        return os.path.join(self.dir, name)


@unittest.skipIf(rl_canvas is None, "reportlab not installed")
class TestValidation(TmpDirCase):
    def test_valid_pdf(self):
        make_text_pdf(self.p("ok.pdf"), [["Hello world"], ["Second page"]])
        r = pp.validate_pdf(self.p("ok.pdf"))
        self.assertTrue(r.ok)
        self.assertEqual(r.page_count, 2)

    def test_missing_file(self):
        self.assertEqual(pp.validate_pdf(self.p("nope.pdf")).error_code, pp.FILE_NOT_FOUND)

    def test_empty_file(self):
        open(self.p("e.pdf"), "wb").close()
        self.assertEqual(pp.validate_pdf(self.p("e.pdf")).error_code, pp.EMPTY_FILE)

    def test_wrong_type(self):
        Path(self.p("a.pdf")).write_text("just some text, not a pdf")
        self.assertEqual(pp.validate_pdf(self.p("a.pdf")).error_code, pp.NOT_A_PDF)

    def test_corrupt(self):
        Path(self.p("c.pdf")).write_bytes(b"%PDF-1.4\nthis is garbage, not a document")
        self.assertEqual(pp.validate_pdf(self.p("c.pdf")).error_code, pp.CORRUPT)

    def test_too_large(self):
        make_text_pdf(self.p("big.pdf"), [["x"]])
        cfg = pp.PipelineConfig(max_file_mb=0.000001)
        self.assertEqual(pp.validate_pdf(self.p("big.pdf"), cfg).error_code, pp.TOO_LARGE)

    def test_too_many_pages(self):
        make_text_pdf(self.p("m.pdf"), [["1"], ["2"], ["3"]])
        cfg = pp.PipelineConfig(max_pages=2)
        r = pp.validate_pdf(self.p("m.pdf"), cfg)
        self.assertEqual(r.error_code, pp.TOO_MANY_PAGES)
        self.assertEqual(r.page_count, 3)

    @unittest.skipIf(PdfWriter is None, "pypdf not installed")
    def test_encrypted(self):
        make_text_pdf(self.p("plain.pdf"), [["secret text"]])
        w = PdfWriter()
        w.append(PdfReader(self.p("plain.pdf")))
        w.encrypt("pw")
        with open(self.p("enc.pdf"), "wb") as fh:
            w.write(fh)
        self.assertEqual(pp.validate_pdf(self.p("enc.pdf")).error_code, pp.ENCRYPTED)

    def test_process_pdf_raises_on_invalid(self):
        with self.assertRaises(pp.PDFValidationError) as ctx:
            pp.process_pdf(self.p("missing.pdf"))
        self.assertEqual(ctx.exception.code, pp.FILE_NOT_FOUND)


class TestFlagsAndRouting(unittest.TestCase):
    def test_garbled_ratio(self):
        self.assertEqual(pp.garbled_ratio("Normal English text."), 0.0)
        self.assertGreater(pp.garbled_ratio("\ufffd\ufffd\ufffd ab"), 0.5)
        self.assertGreater(pp.garbled_ratio("(cid:12)(cid:45)(cid:7) ok"), 0.5)

    def test_decide_route(self):
        cfg = pp.PipelineConfig()
        good = {"char_count": 500, "image_coverage": 0.1, "garbled_ratio": 0.0}
        scan = {"char_count": 0, "image_coverage": 1.0, "garbled_ratio": 0.0}
        blank = {"char_count": 0, "image_coverage": 0.0, "garbled_ratio": 0.0}
        bad = {"char_count": 300, "image_coverage": 1.0, "garbled_ratio": 0.6}
        self.assertEqual(pp.decide_route(good, cfg)[0], "text_layer")
        self.assertEqual(pp.decide_route(scan, cfg)[0], "ocr")
        self.assertEqual(pp.decide_route(blank, cfg)[0], "empty")
        self.assertEqual(pp.decide_route(bad, cfg)[0], "ocr")


@unittest.skipIf(rl_canvas is None, "reportlab not installed")
class TestPipeline(TmpDirCase):
    def test_digital_pdf_extraction_and_contract(self):
        make_text_pdf(self.p("d.pdf"), [
            ["Introduction to document processing", "This page has a valid text layer with enough characters."],
            ["Second page also has plenty of readable English text for extraction."]])
        res = pp.process_pdf(self.p("d.pdf"), doc_id="doc1")
        self.assertEqual(res["page_count"], 2)
        pg = res["pages"][0]
        for key in ("doc_id", "pdf_page_index", "printed_page_label", "source",
                    "raw_text", "text", "search_text", "quality"):
            self.assertIn(key, pg)
        self.assertEqual(pg["doc_id"], "doc1")
        self.assertEqual(pg["pdf_page_index"], 1)
        self.assertEqual(pg["source"]["type"], "text_layer")
        self.assertEqual(pg["source"]["engine"], "pypdfium2")
        self.assertIn("valid text layer", pg["raw_text"])
        self.assertEqual(pg["quality"]["route"], "text_layer")
        self.assertEqual(pg["quality"]["language_guess"], "en")
        json.dumps(res)  # must be JSON-serialisable

    def test_blank_page_flagged_empty(self):
        make_text_pdf(self.p("b.pdf"), [[]])
        pg = pp.process_pdf(self.p("b.pdf"))["pages"][0]
        self.assertIn("empty", pg["quality"]["flags"])
        self.assertEqual(pg["quality"]["route"], "empty")

    def test_scanned_page_routed_to_ocr(self):
        make_scanned_pdf(self.p("s.pdf"), ["Scanned page"])
        cfg = pp.PipelineConfig(ocr_enabled=False)
        pg = pp.process_pdf(self.p("s.pdf"), config=cfg)["pages"][0]
        self.assertEqual(pg["quality"]["route"], "ocr")
        self.assertIn("ocr_skipped", pg["quality"]["flags"])

    @unittest.skipUnless(HAS_TESSERACT, "Tesseract not installed")
    def test_ocr_on_scanned_pdf(self):
        make_scanned_pdf(self.p("s.pdf"), ["Invoice total amount", "Payment due Friday"])
        pg = pp.process_pdf(self.p("s.pdf"))["pages"][0]
        self.assertEqual(pg["source"]["type"], "ocr")
        self.assertEqual(pg["source"]["engine"], "tesseract")
        self.assertIn("invoice", pg["search_text"])
        self.assertIn("payment", pg["search_text"])
        self.assertIsNotNone(pg["quality"]["mean_ocr_confidence"])

    @unittest.skipUnless(HAS_TESSERACT, "Tesseract not installed")
    def test_ocr_on_skewed_scan(self):
        make_scanned_pdf(self.p("k.pdf"), ["Quarterly report summary", "Revenue increased steadily"], angle=3.0)
        pg = pp.process_pdf(self.p("k.pdf"))["pages"][0]
        self.assertIn("quarterly", pg["search_text"])

    def test_short_pdf_pages_keep_content(self):
        make_text_pdf(self.p("short.pdf"), [
            ["Quarterly review", f"Table {i} totals", f"Step {i} complete"] for i in range(1, 7)])
        res = pp.process_pdf(self.p("short.pdf"))
        for i, pg in enumerate(res["pages"], start=1):
            self.assertIn(f"Table {i} totals", pg["text"])
            self.assertIn(f"Step {i} complete", pg["text"])
            self.assertEqual(pg["quality"]["header_footer_lines_removed"], 0)

    def test_json_round_trip(self):
        make_text_pdf(self.p("j.pdf"), [["Round trip text with enough characters to be valid."]])
        res = pp.process_pdf(self.p("j.pdf"))
        pp.save_result_json(res, self.p("out.json"))
        self.assertEqual(pp.load_result_json(self.p("out.json"))["pages"][0]["raw_text"],
                         res["pages"][0]["raw_text"])


class TestPreprocessing(unittest.TestCase):
    def test_skew_estimate_and_preprocess(self):
        img = Image.new("RGB", (1200, 900), "white")
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=40)
        for i in range(12):
            d.text((80, 60 + i * 65), "The quick brown fox jumps over the lazy dog", fill="black", font=font)
        skewed = img.rotate(3.0, resample=Image.BICUBIC, fillcolor="white")
        angle = pp.estimate_skew_angle(skewed)
        self.assertAlmostEqual(angle, -3.0, delta=1.0)
        out = pp.preprocess_image(skewed)
        self.assertEqual(out.mode, "L")
        self.assertTrue(set(out.tobytes()) <= {0, 255})


class TestCleaning(unittest.TestCase):
    def test_unicode_and_ligatures(self):
        t = pp.normalize_unicode_text("o\ufb03ce \ufb01le\u00a0x\u200b y\r\nz\u00ad")
        self.assertEqual(t, "office file x y\nz")

    def test_collapse_whitespace(self):
        self.assertEqual(pp.collapse_whitespace_text("a   b \t c  \n\n\n\nd"), "a b c\n\nd")

    def test_dehyphenate(self):
        self.assertEqual(pp.dehyphenate_text("informa-\ntion is key"), "information is key")
        # next fragment starts uppercase: keep the hyphen/line break
        self.assertEqual(pp.dehyphenate_text("Smith-\nJones"), "Smith-\nJones")

    def test_rejoin_only_long_prose_lines(self):
        long_line = "This sentence is long enough to be a wrapped prose line and"
        self.assertEqual(pp.rejoin_lines_text(long_line + "\ncontinues here."), long_line + " continues here.")
        # short lines (bullets, titles, slide text) stay separate
        self.assertEqual(pp.rejoin_lines_text("Agenda\nitems today"), "Agenda\nitems today")

    def test_header_footer_and_page_numbers_removed_raw_preserved(self):
        pages = []
        topics = ["alpha", "bravo", "charlie", "delta", "echo"]
        for i in range(1, 6):
            raw = f"ACME Corp Annual Report\nBody text of page {topics[i - 1]} with some content.\nPage {i} of 5"
            pages.append({"raw_text": raw, "quality": {}})
        originals = [p["raw_text"] for p in pages]
        pp.clean_pages(pages, pp.CleaningConfig(remove_header_footer=True))   # feature is opt-in
        for i, p in enumerate(pages, start=1):
            self.assertEqual(p["raw_text"], originals[i - 1])          # untouched
            self.assertNotIn("ACME", p["text"])
            self.assertNotIn("of 5", p["text"])
            self.assertIn(f"Body text of page {topics[i - 1]}", p["text"])
            self.assertEqual(p["printed_page_label"], str(i))
            self.assertEqual(p["quality"]["header_footer_lines_removed"], 1)

    def test_header_footer_removal_off_by_default(self):
        self.assertFalse(pp.CleaningConfig().remove_header_footer)
        pages = []
        for i in range(1, 6):
            raw = f"ACME Corp Annual Report\nBody text of page {i} with some content.\nPage {i} of 5"
            pages.append({"raw_text": raw, "quality": {}})
        pp.clean_pages(pages)
        for i, p in enumerate(pages, start=1):
            self.assertIn("ACME Corp Annual Report", p["text"])        # repeated header kept
            self.assertEqual(p["quality"]["header_footer_lines_removed"], 0)
            # page-number extraction is unchanged
            self.assertEqual(p["printed_page_label"], str(i))
            self.assertNotIn("of 5", p["text"])

    def test_short_pages_keep_all_content_by_default(self):
        # Lines differ only by digits, so they look "repeated" once digits are normalised.
        pages = [{"raw_text": f"Quarterly review\nTable {i} totals\nStep {i} complete", "quality": {}}
                 for i in range(1, 7)]
        pp.clean_pages(pages)
        for i, p in enumerate(pages, start=1):
            self.assertEqual(p["text"], f"Quarterly review\nTable {i} totals\nStep {i} complete")

    def test_search_text_and_switches(self):
        pages = [{"raw_text": "Hello   WORLD\nsecond line", "quality": {}}]
        cfg = pp.CleaningConfig(collapse_whitespace=False)
        pp.clean_pages(pages, cfg)
        self.assertIn("   ", pages[0]["text"])                           # rule switched off
        self.assertEqual(pages[0]["search_text"], "hello world second line")
        pages = [{"raw_text": "Keep", "quality": {}}]
        pp.clean_pages(pages, pp.CleaningConfig(make_search_text=False))
        self.assertIsNone(pages[0]["search_text"])

    def test_short_document_keeps_repeated_lines(self):
        pages = [{"raw_text": "Title\nbody one", "quality": {}}, {"raw_text": "Title\nbody two", "quality": {}}]
        pp.clean_pages(pages)
        self.assertIn("Title", pages[0]["text"])                         # below min page count


class FailingImagePage:
    """Stand-in page whose image objects cannot be read."""
    def get_size(self):
        return (100, 100)

    def get_objects(self, **kwargs):
        raise RuntimeError("cannot read objects")


class ZeroSizePage:
    def get_size(self):
        return (0, 0)


class TestImageDetectionFailure(TmpDirCase):
    def test_coverage_is_none_when_objects_unreadable(self):
        self.assertIsNone(pp._image_coverage(FailingImagePage()))

    def test_coverage_is_none_for_zero_area(self):
        self.assertIsNone(pp._image_coverage(ZeroSizePage()))

    def test_unknown_coverage_routes_to_ocr(self):
        cfg = pp.PipelineConfig()
        no_text = {"char_count": 0, "image_coverage": None, "garbled_ratio": 0.0}
        self.assertEqual(pp.decide_route(no_text, cfg),
                         ("ocr", "no_text_layer_image_check_failed"))
        # a page with a good text layer does not need the image check
        good = {"char_count": 500, "image_coverage": None, "garbled_ratio": 0.0}
        self.assertEqual(pp.decide_route(good, cfg)[0], "text_layer")

    def test_pipeline_flags_detection_failure(self):
        make_scanned_pdf(self.p("s.pdf"), ["Scanned page"])
        cfg = pp.PipelineConfig(ocr_enabled=False)
        with mock.patch.object(pp, "_image_coverage", return_value=None):
            res = pp.process_pdf(self.p("s.pdf"), config=cfg)
        pg = res["pages"][0]
        self.assertIn("image_detection_failed", pg["quality"]["flags"])
        self.assertEqual(pg["quality"]["route"], "ocr")      # not silently "empty"
        self.assertIsNone(pg["quality"]["image_coverage"])
        json.dumps(res)


class TestOcrFallback(TmpDirCase):
    """OCR is stubbed, so these tests run without Tesseract."""
    GARBLED = "(cid:1)(cid:2) readable fallback text from the text layer"

    def run_page(self, ocr_text, raw_text):
        make_scanned_pdf(self.p("s.pdf"), ["Scanned page"])
        real_extract = pp.extract_page

        def fake_extract(page):
            flags = real_extract(page)
            flags["raw_text"] = raw_text
            flags["char_count"] = len(raw_text.strip())
            flags["garbled_ratio"] = 0.5 if raw_text else 0.0   # force the OCR route
            return flags

        ocr_result = {"text": ocr_text, "mean_confidence": 90.0 if ocr_text.strip() else None,
                      "word_count": len(ocr_text.split())}
        with mock.patch.object(pp, "tesseract_version", return_value="test"), \
                mock.patch.object(pp, "extract_page", fake_extract), \
                mock.patch.object(pp, "ocr_page", return_value=ocr_result):
            return pp.process_pdf(self.p("s.pdf"))["pages"][0]

    def test_ocr_text_used_and_raw_text_preserved(self):
        pg = self.run_page("Invoice total amount", self.GARBLED)
        self.assertEqual(pg["source"]["type"], "ocr")
        self.assertEqual(pg["raw_text"], self.GARBLED)          # original pypdfium2 output
        self.assertEqual(pg["ocr_text"], "Invoice total amount")
        self.assertIn("Invoice total amount", pg["text"])
        self.assertNotIn("cid", pg["text"])

    def test_empty_ocr_falls_back_to_text_layer(self):
        pg = self.run_page("", self.GARBLED)
        self.assertEqual(pg["source"]["type"], "text_layer")
        self.assertIn("ocr_empty_used_text_layer", pg["quality"]["flags"])
        self.assertNotIn("empty", pg["quality"]["flags"])
        self.assertEqual(pg["ocr_text"], "")
        self.assertIn("readable fallback text", pg["text"])

    def test_whitespace_only_ocr_also_falls_back(self):
        pg = self.run_page("  \n \n", self.GARBLED)
        self.assertEqual(pg["source"]["type"], "text_layer")
        self.assertIn("readable fallback text", pg["text"])

    def test_empty_ocr_and_no_text_layer_is_empty(self):
        pg = self.run_page("", "")
        self.assertEqual(pg["source"]["type"], "ocr")
        self.assertIn("empty", pg["quality"]["flags"])
        self.assertNotIn("ocr_empty_used_text_layer", pg["quality"]["flags"])
        self.assertEqual(pg["text"], "")


class TestOcrFlags(TmpDirCase):
    """Documented OCR flags that had no test. OCR and the Tesseract check are stubbed."""

    def scan(self):
        make_scanned_pdf(self.p("s.pdf"), ["Scanned page"])
        return self.p("s.pdf")

    def test_tesseract_missing_flags_page_and_document(self):
        with mock.patch.object(pp, "tesseract_version", return_value=None):
            res = pp.process_pdf(self.scan())
        pg = res["pages"][0]
        self.assertEqual(pg["quality"]["route"], "ocr")
        self.assertIn("ocr_unavailable", pg["quality"]["flags"])
        self.assertEqual(pg["text"], "")
        self.assertTrue(res["warnings"])                  # scanned pages are never silently lost

    def test_low_confidence_ocr_is_flagged_but_kept(self):
        blurry = {"text": "some blurry words here", "mean_confidence": 42.0, "word_count": 4}
        with mock.patch.object(pp, "tesseract_version", return_value="test"), \
                mock.patch.object(pp, "ocr_page", return_value=blurry):
            pg = pp.process_pdf(self.scan())["pages"][0]
        self.assertIn("low_ocr_confidence", pg["quality"]["flags"])
        self.assertEqual(pg["source"]["type"], "ocr")
        self.assertIn("blurry words", pg["text"])

    def test_ocr_exception_is_flagged_not_fatal(self):
        with mock.patch.object(pp, "tesseract_version", return_value="test"), \
                mock.patch.object(pp, "ocr_page", side_effect=RuntimeError("boom")):
            res = pp.process_pdf(self.scan())
        pg = res["pages"][0]
        self.assertIn("ocr_failed", pg["quality"]["flags"])
        self.assertTrue(any("boom" in w for w in pg["quality"]["warnings"]))
        self.assertEqual(pg["text"], "")
        json.dumps(res)

    def test_non_english_script_is_flagged(self):
        arabic = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645 \u0641\u064a \u0647\u0630\u0627 \u0627\u0644\u0645\u0633\u062a\u0646\u062f " * 4
        real_extract = pp.extract_page

        def fake_extract(page):
            flags = real_extract(page)
            flags.update(raw_text=arabic, char_count=len(arabic), garbled_ratio=0.0)
            return flags

        with mock.patch.object(pp, "extract_page", fake_extract):
            pg = pp.process_pdf(self.scan())["pages"][0]
        self.assertIn("non_english_script", pg["quality"]["flags"])
        self.assertEqual(pg["quality"]["language_guess"], "non_en")


SAMPLE_PDF = Path(__file__).with_name("sample_scanned_page.pdf")
# Manual transcription of the sample page (the signature scribble is not text).
SAMPLE_REFERENCE = """LARKSPUR COURT RESIDENTS' ASSOCIATION
Fictional association - 1 Specimen Lane, Sampleton 000 000
NOTICE OF PLANNED MAINTENANCE Date: 6 October 2026
Dear residents,
The water tanks and the main stairwell lighting in all three blocks will be serviced during the week of 12 October 2026. Water will be turned off in each block for the hours shown below. Please store enough water for those hours and keep the stairwell doors clear so that the contractor can work safely.
The lifts will run as normal. If you have a delivery booked for one of these days, please tell the caretaker by Friday 9 October so that the side gate can be left open.
Block Day Water off Work
A Mon 12 Oct 09:00 - 13:00 Tank cleaning
B Tue 13 Oct 09:00 - 13:00 Tank cleaning
C Wed 14 Oct 10:00 - 15:00 Tank cleaning and pump check
All Thu 15 Oct none Stairwell lighting
Questions about this work can be sent to the association office by letter or left with the caretaker. Thank you for your patience.
Yours sincerely,
Sam Specimen
Secretary, Larkspur Court Residents' Association (fictional)
SPECIMEN - fictional document made by filedone.ai for testing. Free to use (CC0 1.0). Page 1 of 1"""


def _similarity(reference, text):
    """difflib similarity of two texts after whitespace and apostrophe normalisation."""
    def norm(t):
        return " ".join(t.replace("\u2019", "'").split())
    return difflib.SequenceMatcher(None, norm(reference), norm(text), autojunk=False).ratio()


@unittest.skipUnless(HAS_TESSERACT, "Tesseract not installed")
class TestSampleScannedPdf(unittest.TestCase):
    """A real scanned English page (fictional notice, CC0) through the full pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.result = pp.process_pdf(str(SAMPLE_PDF))
        cls.page = cls.result["pages"][0]

    def test_runs_cleanly_through_ocr(self):
        self.assertEqual(self.result["page_count"], 1)
        self.assertEqual(self.result["warnings"], [])
        self.assertEqual(self.page["raw_text"], "")                  # no text layer
        self.assertEqual(self.page["quality"]["route"], "ocr")
        self.assertEqual(self.page["source"]["type"], "ocr")
        self.assertEqual(self.page["quality"]["flags"], [])
        self.assertGreaterEqual(self.page["quality"]["mean_ocr_confidence"], 90)

    def test_key_facts_are_extracted(self):
        text = self.page["text"]
        for fact in ("LARKSPUR COURT", "6 October 2026", "week of 12 October 2026",
                     "Friday 9 October", "Wed 14 Oct", "10:00 - 15:00",
                     "Tank cleaning and pump check", "Sam Specimen"):
            self.assertIn(fact, text)

    def test_text_is_close_to_reference(self):
        # about 0.995 measured; margin for Tesseract versions
        self.assertGreaterEqual(_similarity(SAMPLE_REFERENCE, self.page["text"]), 0.97)


@unittest.skipUnless(HAS_TESSERACT, "Tesseract not installed")
class TestDegradedScannedPdf(unittest.TestCase):
    """Same page, degraded: rotated 2 degrees, about 120 DPI, blurred, noisy, JPEG quality 35."""

    @classmethod
    def setUpClass(cls):
        cls.result = pp.process_pdf(str(Path(__file__).with_name("sample_scanned_page_degraded.pdf")))
        cls.page = cls.result["pages"][0]

    def test_runs_cleanly_through_ocr(self):
        self.assertEqual(self.result["warnings"], [])
        self.assertEqual(self.page["quality"]["route"], "ocr")
        self.assertEqual(self.page["quality"]["flags"], [])
        self.assertGreaterEqual(self.page["quality"]["mean_ocr_confidence"], 75)   # about 87 measured

    def test_body_text_survives_degradation(self):
        text = self.page["text"]
        for fact in ("LARKSPUR COURT", "6 October 2026", "week of 12 October 2026",
                     "Friday 9 October", "10:00 - 15:00", "Tank cleaning and pump check",
                     "Sam Specimen"):
            self.assertIn(fact, text)

    def test_text_is_still_close_to_reference(self):
        # about 0.96 measured; tiny footer text and one table cell are damaged
        self.assertGreaterEqual(_similarity(SAMPLE_REFERENCE, self.page["text"]), 0.93)


@unittest.skipUnless(HAS_TESSERACT, "Tesseract not installed")
class TestMixedDigitalAndScannedPdf(unittest.TestCase):
    """Page 1 is a digital text page, page 2 is the scanned page: each must take its own route."""

    @classmethod
    def setUpClass(cls):
        cls.result = pp.process_pdf(str(Path(__file__).with_name("sample_mixed_digital_scanned.pdf")))
        cls.digital, cls.scanned = cls.result["pages"]

    def test_digital_page_uses_text_layer_without_ocr(self):
        q = self.digital["quality"]
        self.assertEqual((q["route"], self.digital["source"]["type"]), ("text_layer", "text_layer"))
        self.assertEqual(self.digital["source"]["engine"], "pypdfium2")
        self.assertIsNone(self.digital["ocr_text"])
        self.assertIsNone(q["mean_ocr_confidence"])
        for fact in ("MINUTES OF THE COMMITTEE MEETING", "29 September 2026", "4,250",
                     "Tuesday 24 November 2026 at 19:00"):
            self.assertIn(fact, self.digital["raw_text"])

    def test_scanned_page_uses_ocr(self):
        q = self.scanned["quality"]
        self.assertEqual((q["route"], self.scanned["source"]["type"]), ("ocr", "ocr"))
        self.assertEqual(self.scanned["raw_text"], "")
        self.assertIsNotNone(self.scanned["ocr_text"])
        self.assertGreaterEqual(q["mean_ocr_confidence"], 90)
        self.assertGreaterEqual(_similarity(SAMPLE_REFERENCE, self.scanned["text"]), 0.97)

    def test_document_level_result(self):
        self.assertEqual(self.result["page_count"], 2)
        self.assertEqual([p["pdf_page_index"] for p in self.result["pages"]], [1, 2])
        self.assertEqual(self.result["warnings"], [])
        self.assertEqual([p["quality"]["flags"] for p in self.result["pages"]], [[], []])


if __name__ == "__main__":
    unittest.main()
