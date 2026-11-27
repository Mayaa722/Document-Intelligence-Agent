"""Tests for chunking.py.

Run from the repository root with either:
    python -m unittest discover -s tests -v
    pytest -q
The core tests use hand-made page dicts and need only the standard library. The tests that
run the real PDF Processing pipeline need reportlab (and, for the scanned page, Tesseract);
they skip automatically when those are missing.
"""

import copy
import os
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chunking as ck  # noqa: E402

try:
    import pdf_processing as pp
except ImportError:  # pragma: no cover
    pp = None
try:
    from reportlab.pdfgen import canvas as rl_canvas
except ImportError:  # pragma: no cover
    rl_canvas = None

HAS_PIPELINE = pp is not None and rl_canvas is not None
HAS_TESSERACT = pp is not None and pp.tesseract_version() is not None
MIXED_PDF = Path(__file__).with_name("sample_mixed_digital_scanned.pdf")


def make_page(index, text, label=None, doc_id="doc1", source_type="text_layer", flags=None):
    """A page dict with the fields chunking.py reads (same shape as pdf_processing output)."""
    return {
        "doc_id": doc_id, "pdf_page_index": index, "printed_page_label": label,
        "source": {"type": source_type}, "raw_text": text, "ocr_text": None, "text": text,
        "search_text": text.lower(),
        "quality": {"flags": list(flags or []), "warnings": []},
    }


def make_result(pages, doc_id="doc1"):
    return {"doc_id": doc_id, "source_file": f"{doc_id}.pdf", "page_count": len(pages),
            "config": {}, "pages": pages, "warnings": []}


def spans_text(text, cfg, **kw):
    return [text[s:e] for s, e in ck.split_text(text, cfg, **kw)]


def random_text(seed, paragraphs=12):
    rng = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "report", "evidence", "page", "chunk",
             "water", "tank", "committee", "maintenance", "budget"]
    paras = []
    for _ in range(rng.randint(paragraphs // 2, paragraphs)):
        lines = []
        for _ in range(rng.randint(1, 4)):
            sentences = []
            for _ in range(rng.randint(1, 5)):
                words = [rng.choice(vocab) for _ in range(rng.randint(3, 15))]
                if rng.random() < 0.05:
                    words.append("z" * rng.randint(100, 300))     # unbroken run
                sentences.append(" ".join(words).capitalize() + ".")
            lines.append(" ".join(sentences))
        paras.append("\n".join(lines))
    return "\n\n".join(paras)


class TestConfig(unittest.TestCase):
    def test_defaults_are_valid_and_provisional(self):
        cfg = ck.ChunkingConfig()
        self.assertEqual((cfg.chunk_size, cfg.chunk_overlap, cfg.length_unit),
                         (1000, 150, "characters"))
        self.assertEqual(cfg.to_dict(), {"chunk_size": 1000, "chunk_overlap": 150,
                                         "length_unit": "characters"})

    def test_invalid_values_rejected(self):
        for kwargs in ({"chunk_size": 0}, {"chunk_size": -5}, {"chunk_overlap": -1},
                       {"chunk_size": 100, "chunk_overlap": 100},
                       {"chunk_size": 100, "chunk_overlap": 150}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                ck.ChunkingConfig(**kwargs)

    def test_validate_catches_later_changes(self):
        cfg = ck.ChunkingConfig(chunk_size=100, chunk_overlap=10)
        cfg.chunk_overlap = 200
        with self.assertRaises(ValueError):
            ck.split_text("some text", cfg)


class TestSplitLevels(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        text = "A short page."
        self.assertEqual(spans_text(text, ck.ChunkingConfig(100, 10)), [text])

    def test_empty_and_whitespace_text(self):
        for text in ("", "   \n\n  \t", None):
            self.assertEqual(ck.split_text(text, ck.ChunkingConfig(100, 10)), [])

    def test_splits_on_paragraphs_first(self):
        text = "Para one is here.\n\nPara two is here.\n\nPara three is here."
        self.assertEqual(spans_text(text, ck.ChunkingConfig(20, 0)),
                         ["Para one is here.", "Para two is here.", "Para three is here."])

    def test_small_paragraphs_are_merged_up_to_size(self):
        text = "One.\n\nTwo.\n\nThree.\n\nFour."
        self.assertEqual(spans_text(text, ck.ChunkingConfig(14, 0)),
                         ["One.\n\nTwo.", "Three.\n\nFour."])

    def test_oversized_paragraph_is_not_mixed_with_the_next_one(self):
        # Paragraph 2 is too long and is split by itself; no chunk mixes the end of
        # paragraph 1 with the start of paragraph 2.
        p1 = "First paragraph here."
        p2 = "Second paragraph is a lot longer and has to be split into several pieces."
        chunks = spans_text(p1 + "\n\n" + p2, ck.ChunkingConfig(40, 0))
        self.assertEqual(chunks[0], p1)
        self.assertTrue(all("First" not in c for c in chunks[1:]))
        self.assertTrue(all(not c.startswith("First") or c == p1 for c in chunks))

    def test_falls_back_to_lines(self):
        text = "line one is here\nline two is here\nline three is here"
        self.assertEqual(spans_text(text, ck.ChunkingConfig(20, 0)),
                         ["line one is here", "line two is here", "line three is here"])

    def test_falls_back_to_sentences(self):
        text = "The first sentence ends here. The second one follows it. A third closes it."
        self.assertEqual(spans_text(text, ck.ChunkingConfig(32, 0)),
                         ["The first sentence ends here.", "The second one follows it.",
                          "A third closes it."])

    def test_sentence_split_ignores_decimal_points(self):
        text = "The rate is 3.5 percent. It was 2.1 before."
        chunks = spans_text(text, ck.ChunkingConfig(26, 0))
        self.assertEqual(chunks, ["The rate is 3.5 percent.", "It was 2.1 before."])

    def test_falls_back_to_words(self):
        text = "one two three four five six seven eight nine ten"
        chunks = spans_text(text, ck.ChunkingConfig(15, 0))
        self.assertEqual(chunks, ["one two three", "four five six", "seven eight", "nine ten"])

    def test_hard_cut_for_unbroken_run(self):
        text = "x" * 25
        self.assertEqual(spans_text(text, ck.ChunkingConfig(10, 0)),
                         ["x" * 10, "x" * 10, "x" * 5])

    def test_long_token_between_words(self):
        text = "before " + "y" * 30 + " after"
        chunks = spans_text(text, ck.ChunkingConfig(12, 0))
        self.assertTrue(all(len(c) <= 12 for c in chunks))
        self.assertEqual("".join(chunks).replace(" ", ""), text.replace(" ", ""))


class TestInvariants(unittest.TestCase):
    """Properties that must hold for any text and any valid configuration."""

    CONFIGS = [(80, 0), (80, 20), (200, 50), (500, 100), (1000, 150), (30, 29), (12, 5)]

    def test_size_offsets_coverage_progress_and_overlap(self):
        for seed in range(8):
            text = random_text(seed)
            for size, overlap in self.CONFIGS:
                cfg = ck.ChunkingConfig(size, overlap)
                spans = ck.split_text(text, cfg)
                tag = f"seed={seed} size={size} overlap={overlap}"
                covered = [False] * len(text)
                prev = None
                for s, e in spans:
                    piece = text[s:e]
                    self.assertLessEqual(len(piece), size, tag)
                    self.assertEqual(piece, piece.strip(), tag)
                    self.assertTrue(piece, tag)
                    for k in range(s, e):
                        covered[k] = True
                    if prev:
                        self.assertGreater(s, prev[0], tag)         # starts move forward
                        self.assertGreater(e, prev[1], tag)         # ends move forward
                        self.assertLessEqual(max(0, prev[1] - s), overlap, tag)
                    prev = (s, e)
                missing = [k for k, ch in enumerate(text) if not ch.isspace() and not covered[k]]
                self.assertEqual(missing, [], tag)                  # no text is lost

    def test_zero_overlap_chunks_do_not_overlap(self):
        text = random_text(3)
        spans = ck.split_text(text, ck.ChunkingConfig(120, 0))
        for (_, e1), (s2, _) in zip(spans, spans[1:]):
            self.assertLessEqual(e1, s2)

    def test_no_chunk_is_only_overlap(self):
        # A split paragraph followed by a blank line used to leave a whitespace-only piece
        # that produced a chunk containing nothing but repeated text.
        text = ("A long first paragraph with several short words in it.\n\n"
                "Second paragraph.\n\nThird one here.")
        spans = ck.split_text(text, ck.ChunkingConfig(30, 12))
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            self.assertGreater(e2, e1)
            self.assertGreater(s2, s1)

    def test_deterministic(self):
        text = random_text(5)
        cfg = ck.ChunkingConfig(150, 30)
        self.assertEqual(ck.split_text(text, cfg), ck.split_text(text, cfg))


class TestOverlap(unittest.TestCase):
    def test_overlap_uses_whole_trailing_pieces(self):
        text = "aaa bbb ccc ddd eee fff ggg hhh"
        chunks = spans_text(text, ck.ChunkingConfig(15, 8))
        self.assertEqual(chunks, ["aaa bbb ccc ddd", "ccc ddd eee fff", "fff ggg hhh"])

    def test_overlap_is_zero_when_last_piece_is_longer_than_overlap(self):
        text = "Para one is quite long here.\n\nPara two is quite long here."
        cfg = ck.ChunkingConfig(32, 10)
        self.assertEqual(spans_text(text, cfg),
                         ["Para one is quite long here.", "Para two is quite long here."])

    def test_overlap_close_to_chunk_size_still_terminates(self):
        text = " ".join(f"w{i}" for i in range(200))
        spans = ck.split_text(text, ck.ChunkingConfig(20, 19))
        self.assertGreater(len(spans), 1)
        self.assertEqual(len(set(spans)), len(spans))


class TestCustomLengthFunction(unittest.TestCase):
    """Chunk size can be measured by any function, so a tokenizer can be plugged in later."""

    def test_word_count_as_length(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        count = lambda s: len(s.split())          # noqa: E731
        cfg = ck.ChunkingConfig(chunk_size=5, chunk_overlap=0, length_unit="words")
        chunks = spans_text(text, cfg, length_function=count)
        self.assertEqual(chunks, ["one two three four five", "six seven eight nine ten",
                                  "eleven twelve"])
        self.assertTrue(all(count(c) <= 5 for c in chunks))

    def test_length_is_reported_in_the_same_unit(self):
        count = lambda s: len(s.split())          # noqa: E731
        cfg = ck.ChunkingConfig(chunk_size=4, chunk_overlap=0, length_unit="words")
        out = ck.chunk_document(make_result([make_page(1, "a b c d e f g")]), cfg,
                                length_function=count)
        self.assertEqual([c["length"] for c in out["chunks"]], [4, 3])
        self.assertEqual(out["config"]["length_unit"], "words")

    def test_default_length_is_characters(self):
        out = ck.chunk_document(make_result([make_page(1, "hello world")]))
        self.assertEqual(out["chunks"][0]["length"], 11)
        self.assertEqual(out["config"]["length_unit"], "characters")


class TestMetadataAndPageBoundaries(unittest.TestCase):
    def setUp(self):
        self.p1 = "Alpha one is here.\n\nAlpha two is here.\n\nAlpha three is here."
        self.p2 = "Beta one is here.\n\nBeta two is here."
        self.p3 = "Gamma only."
        self.result = make_result([
            make_page(1, self.p1, label=None, doc_id="report"),
            make_page(2, self.p2, label="7", doc_id="report", source_type="ocr",
                      flags=["low_ocr_confidence"]),
            make_page(3, self.p3, label="8", doc_id="report"),
        ], doc_id="report")
        self.cfg = ck.ChunkingConfig(chunk_size=45, chunk_overlap=0)
        self.out = ck.chunk_document(self.result, self.cfg)
        self.chunks = self.out["chunks"]

    def test_chunks_never_contain_text_from_another_page(self):
        marker = {1: "Alpha", 2: "Beta", 3: "Gamma"}
        for ch in self.chunks:
            for page, word in marker.items():
                if page != ch["pdf_page_index"]:
                    self.assertNotIn(word, ch["text"])

    def test_a_short_page_is_not_merged_with_its_neighbour(self):
        # Pages 2 and 3 together would fit in one chunk, but they are different pages.
        self.assertEqual([c["pdf_page_index"] for c in self.chunks if "Gamma" in c["text"]], [3])
        self.assertEqual(sorted({c["pdf_page_index"] for c in self.chunks}), [1, 2, 3])

    def test_every_chunk_lies_inside_its_own_page_text(self):
        pages = {p["pdf_page_index"]: p["text"] for p in self.result["pages"]}
        for ch in self.chunks:
            page_text = pages[ch["pdf_page_index"]]
            self.assertEqual(page_text[ch["char_start"]:ch["char_end"]], ch["text"])

    def test_metadata_is_preserved(self):
        for ch in self.chunks:
            self.assertEqual(ch["doc_id"], "report")
        by_page = {}
        for ch in self.chunks:
            by_page.setdefault(ch["pdf_page_index"], []).append(ch)
        self.assertTrue(all(c["printed_page_label"] is None for c in by_page[1]))
        self.assertTrue(all(c["printed_page_label"] == "7" for c in by_page[2]))
        self.assertTrue(all(c["printed_page_label"] == "8" for c in by_page[3]))
        self.assertTrue(all(c["source_type"] == "ocr" and c["flags"] == ["low_ocr_confidence"]
                            for c in by_page[2]))
        self.assertTrue(all(c["source_type"] == "text_layer" and c["flags"] == []
                            for c in by_page[1]))

    def test_chunk_index_runs_over_the_document_and_page_index_restarts(self):
        self.assertEqual([c["chunk_index"] for c in self.chunks], list(range(len(self.chunks))))
        for page in (1, 2, 3):
            idx = [c["page_chunk_index"] for c in self.chunks if c["pdf_page_index"] == page]
            self.assertEqual(idx, list(range(len(idx))))

    def test_chunk_ids_are_unique_and_readable(self):
        ids = [c["chunk_id"] for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "report_p1_c0")

    def test_document_fields(self):
        self.assertEqual(self.out["doc_id"], "report")
        self.assertEqual(self.out["source_file"], "report.pdf")
        self.assertEqual(self.out["page_count"], 3)
        self.assertEqual(self.out["config"]["chunk_size"], 45)
        self.assertEqual(self.out["warnings"], [])

    def test_pages_are_chunked_in_input_order(self):
        reordered = make_result([self.result["pages"][2], self.result["pages"][0]],
                                doc_id="report")
        out = ck.chunk_document(reordered, self.cfg)
        self.assertEqual([c["pdf_page_index"] for c in out["chunks"]][0], 3)
        self.assertEqual(out["chunks"][0]["chunk_index"], 0)

    def test_empty_page_gives_warning_and_no_chunks(self):
        result = make_result([make_page(1, "Real text."), make_page(2, "", flags=["empty"]),
                              make_page(3, "More text.")])
        out = ck.chunk_document(result, ck.ChunkingConfig(100, 10))
        self.assertEqual([c["pdf_page_index"] for c in out["chunks"]], [1, 3])
        self.assertEqual([c["chunk_index"] for c in out["chunks"]], [0, 1])
        self.assertEqual(len(out["warnings"]), 1)
        self.assertIn("Page 2", out["warnings"][0])

    def test_page_with_missing_text_is_treated_as_empty(self):
        page = make_page(1, "x")
        page["text"] = None
        out = ck.chunk_document(make_result([page]))
        self.assertEqual(out["chunks"], [])
        self.assertEqual(len(out["warnings"]), 1)

    def test_overlap_stays_inside_one_page(self):
        text = " ".join(f"word{i}" for i in range(60))
        result = make_result([make_page(1, text), make_page(2, text)])
        out = ck.chunk_document(result, ck.ChunkingConfig(60, 20))
        pages = {1: text, 2: text}
        for ch in out["chunks"]:
            self.assertEqual(pages[ch["pdf_page_index"]][ch["char_start"]:ch["char_end"]],
                             ch["text"])
        first_of_page2 = [c for c in out["chunks"] if c["pdf_page_index"] == 2][0]
        self.assertEqual(first_of_page2["char_start"], 0)     # no overlap carried over from page 1
        self.assertEqual(first_of_page2["page_chunk_index"], 0)

    def test_input_is_not_modified(self):
        before = copy.deepcopy(self.result)
        ck.chunk_document(self.result, self.cfg)
        self.assertEqual(self.result, before)

    def test_text_is_the_only_field_chunked(self):
        page = make_page(1, "Cleaned text only.")
        page["raw_text"] = "RAW TEXT THAT MUST NOT BE USED"
        page["ocr_text"] = "OCR TEXT THAT MUST NOT BE USED"
        out = ck.chunk_document(make_result([page]))
        self.assertEqual([c["text"] for c in out["chunks"]], ["Cleaned text only."])

    def test_chunk_pages_accepts_pages_without_a_result_wrapper(self):
        chunks, warnings = ck.chunk_pages([make_page(1, "Hello there.")], ck.ChunkingConfig(50, 5))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(warnings, [])


class TestJson(unittest.TestCase):
    def test_round_trip(self):
        out = ck.chunk_document(make_result([make_page(1, "Caf\u00e9 \u2013 r\u00e9sum\u00e9 text. " * 30)]),
                                ck.ChunkingConfig(100, 20))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chunks.json")
            ck.save_chunks_json(out, path)
            self.assertEqual(ck.load_chunks_json(path), out)


@unittest.skipUnless(HAS_PIPELINE, "pdf_processing or reportlab not available")
class TestWithPdfProcessing(unittest.TestCase):
    """Chunking the real output of pdf_processing.process_pdf."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def make_pdf(self, name, pages):
        path = os.path.join(self.dir, name)
        c = rl_canvas.Canvas(path)
        for lines in pages:
            y = 780
            for line in lines:
                c.drawString(60, y, line)
                y -= 18
            c.showPage()
        c.save()
        return path

    def test_digital_pdf_end_to_end(self):
        def body(word, n):
            return [f"This {word} sentence number {i} describes the {word} material in detail."
                    for i in range(n)]
        path = self.make_pdf("doc.pdf", [body("alpha", 14) + ["Page 1"],
                                         body("beta", 14) + ["Page 2"],
                                         body("gamma", 3) + ["Page 3"]])
        result = pp.process_pdf(path, doc_id="doc")
        cfg = ck.ChunkingConfig(chunk_size=300, chunk_overlap=60)
        out = ck.chunk_document(result, cfg)
        chunks = out["chunks"]

        self.assertEqual(out["doc_id"], "doc")
        self.assertEqual(sorted({c["pdf_page_index"] for c in chunks}), [1, 2, 3])
        self.assertEqual([c["chunk_index"] for c in chunks], list(range(len(chunks))))
        self.assertGreater(len([c for c in chunks if c["pdf_page_index"] == 1]), 1)
        pages = {p["pdf_page_index"]: p for p in result["pages"]}
        for ch in chunks:
            page = pages[ch["pdf_page_index"]]
            self.assertEqual(ch["doc_id"], "doc")
            self.assertEqual(ch["printed_page_label"], page["printed_page_label"])
            self.assertEqual(page["text"][ch["char_start"]:ch["char_end"]], ch["text"])
            self.assertLessEqual(len(ch["text"]), 300)
            self.assertEqual(ch["source_type"], "text_layer")
            other = {"alpha", "beta", "gamma"} - {("alpha", "beta", "gamma")[ch["pdf_page_index"] - 1]}
            for word in other:
                self.assertNotIn(word, ch["text"])
        # printed page numbers are taken out of the text by pdf_processing and kept as labels
        self.assertEqual([pages[i]["printed_page_label"] for i in (1, 2, 3)], ["1", "2", "3"])
        self.assertTrue(all(not re.search(r"\bPage \d\b", c["text"]) for c in chunks))

    def test_page_without_text_is_reported(self):
        path = self.make_pdf("blank.pdf", [["Some real text on page one for the test."], []])
        result = pp.process_pdf(path)
        out = ck.chunk_document(result, ck.ChunkingConfig(200, 20))
        self.assertEqual({c["pdf_page_index"] for c in out["chunks"]}, {1})
        self.assertEqual(len(out["warnings"]), 1)
        self.assertIn("Page 2", out["warnings"][0])

    def test_mixed_pdf_without_ocr_still_chunks_the_digital_page(self):
        cfg = pp.PipelineConfig(ocr_enabled=False)
        result = pp.process_pdf(str(MIXED_PDF), config=cfg)
        out = ck.chunk_document(result, ck.ChunkingConfig(300, 50))
        self.assertEqual({c["pdf_page_index"] for c in out["chunks"]}, {1})
        self.assertEqual(len(out["warnings"]), 1)           # the scanned page has no text
        self.assertIn("Page 2", out["warnings"][0])

    @unittest.skipUnless(HAS_TESSERACT, "Tesseract not installed")
    def test_mixed_digital_and_scanned_pdf(self):
        result = pp.process_pdf(str(MIXED_PDF), doc_id="mixed")
        out = ck.chunk_document(result, ck.ChunkingConfig(300, 50))
        chunks = out["chunks"]
        self.assertEqual(out["warnings"], [])
        self.assertEqual(sorted({c["pdf_page_index"] for c in chunks}), [1, 2])
        p1 = [c for c in chunks if c["pdf_page_index"] == 1]
        p2 = [c for c in chunks if c["pdf_page_index"] == 2]
        self.assertTrue(all(c["source_type"] == "text_layer" for c in p1))
        self.assertTrue(all(c["source_type"] == "ocr" for c in p2))
        self.assertIn("MINUTES", " ".join(c["text"] for c in p1))
        self.assertIn("MAINTENANCE", " ".join(c["text"] for c in p2))
        self.assertFalse(any("MINUTES" in c["text"] for c in p2))
        self.assertFalse(any("NOTICE OF PLANNED" in c["text"] for c in p1))
        self.assertEqual([c["chunk_index"] for c in chunks], list(range(len(chunks))))
        self.assertTrue(all(c["doc_id"] == "mixed" for c in chunks))


if __name__ == "__main__":
    unittest.main()
