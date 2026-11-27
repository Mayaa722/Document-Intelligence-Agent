# PDF Processing

Turns an uploaded PDF into page-level JSON for the Document Intelligence Agent: validation, text extraction, OCR for pages that need it, and conservative English cleaning. CPU only, free and open-source tools, no external APIs.

| Step | Method |
|---|---|
| Validation | `%PDF-` signature, open test, size/page limits, structured error codes |
| Extraction | pypdfium2, page by page |
| OCR | Tesseract (English), 300 DPI, deskew and Otsu binarisation, only for pages that need it |
| Cleaning | Conservative English rules; `raw_text` and `ocr_text` are never modified, cleaning only produces `text` and `search_text` |

Chunking is a separate module, `chunking.py` (see [Chunking](#chunking)), that reads this module's output. Retrieval is handled downstream.

## Install

```bash
pip install -r requirements.txt     # runtime packages plus test-only packages
# Tesseract is a system package (not pip). English ("eng") data is required.
sudo apt-get install tesseract-ocr tesseract-ocr-eng     # Ubuntu/Debian/Kaggle
brew install tesseract                                   # macOS (includes eng and osd)
tesseract --list-langs                                   # must list "eng"
```

- **Ubuntu/Debian:** English is a separate language package (`tesseract-ocr-eng`); naming it explicitly is harmless if it is already installed.
- **macOS:** the Homebrew formula contains only `eng` and `osd`. Other languages need `brew install tesseract-lang`.
- **Windows:** use the installer from the [UB Mannheim wiki](https://github.com/UB-Mannheim/tesseract/wiki) and add its folder (for example `C:\Program Files\Tesseract-OCR`) to `PATH`. If `tesseract --list-langs` does not show `eng`, re-run the installer and select the English model, or copy `eng.traineddata` into its `tessdata` folder.

Without Tesseract, everything except OCR still works. Pages that need OCR are flagged `ocr_unavailable` and the result gets a warning, so scanned pages are never silently lost. On Kaggle, turn Internet on for the installs.

## Usage

```python
import pdf_processing as pp

result = pp.process_pdf("lecture1.pdf")     # raises PDFValidationError if the file is rejected
for page in result["pages"]:
    page["text"]          # cleaned text (use for chunking and LLM context)
    page["search_text"]   # lowercase copy for BM25
    page["raw_text"]      # exactly what pypdfium2 returned
    page["ocr_text"]      # Tesseract text, or None if OCR was not used

pp.save_result_json(result, "lecture1.pages.json")
result = pp.load_result_json("lecture1.pages.json")
```

Check a file without processing it (never raises):

```python
check = pp.validate_pdf("file.pdf")
if not check.ok:
    print(check.error_code, check.message)
```

All limits and thresholds live in one config:

```python
cfg = pp.PipelineConfig(max_pages=300, ocr_dpi=200)
cfg.cleaning.dehyphenate = False            # each cleaning rule can be switched off
result = pp.process_pdf("file.pdf", config=cfg)
```

Error codes: `FILE_NOT_FOUND`, `EMPTY_FILE`, `NOT_A_PDF`, `TOO_LARGE`, `CORRUPT`, `ENCRYPTED` (password required), `ZERO_PAGES`, `TOO_MANY_PAGES`.

## How pages are handled

Each page is routed on its text layer and image content:

- Garbled text layer, or almost no text on an image-like page: **OCR**.
- Almost no text and the image check failed: **OCR** (so a detection failure never hides a scan).
- Almost no text and no image: `empty`.
- Otherwise: the **text layer**.

If OCR returns nothing but the text layer has text, the text layer is kept (flag `ocr_empty_used_text_layer`).

## Output

`process_pdf` returns `{doc_id, source_file, page_count, config, pages, warnings}`. Each page:

```text
doc_id, pdf_page_index (1-based, use for citations), printed_page_label (or null)
source        {type: "text_layer" | "ocr", engine, version}: which text was cleaned
raw_text      pypdfium2 output, always unmodified
ocr_text      Tesseract output, or null if OCR was not used
text          cleaned text (from ocr_text when source.type is "ocr", otherwise raw_text)
search_text   lowercase, single-spaced copy (null if switched off)
quality       route, route_reason, language_guess, char_count, image_coverage (null if
              not measurable), garbled_ratio, mean_ocr_confidence, flags, warnings,
              header_footer_lines_removed
```

Flags: `empty`, `garbled`, `low_ocr_confidence`, `ocr_failed`, `ocr_unavailable`, `ocr_skipped`, `ocr_empty_used_text_layer`, `image_detection_failed`, `non_english_script`.

## Cleaning (English only)

Unicode normalisation and ligature expansion, control and zero-width character removal, whitespace collapse, de-hyphenation of line-end splits, re-joining of wrapped prose lines (short lines, bullets and table rows stay separate), and moving standalone page numbers into `printed_page_label`. No stemming, stop-word removal or lowercasing is applied to `text`.

Header/footer removal is **off by default** because it can delete real content, especially on short pages such as slides and forms. Enable it with `cfg.cleaning.remove_header_footer = True` for long documents with real running headers (it needs at least 4 pages).

Limits: thresholds are provisional starting values; non-Latin text (for example Arabic) is flagged but not cleaned; tables are kept line by line.

## OCR sample tests

Three PDFs in `tests/` exercise the real pipeline end to end (Tesseract 5.3.4, default config). All three are covered by classes in `tests/test_pdf_processing.py` and skip if Tesseract is not installed.

| Fixture | What it is | What the tests verify | Result |
|---|---|---|---|
| `sample_scanned_page.pdf` | One-page scanned English notice, image only, no text layer. A fictional CC0 specimen (its footer credits filedone.ai). | Routed to OCR, no flags or warnings, mean confidence >= 90, key facts present, similarity to a manual transcription >= 0.97. | PASS: confidence 94.8, similarity 0.995 (CER 0.8%, WER 2.9%) |
| `sample_scanned_page_degraded.pdf` | The same page rotated 2 degrees, reduced to about 120 DPI, blurred, noisy and JPEG-compressed (quality 35). | Still routed to OCR with no flags, confidence >= 75, key facts present, similarity >= 0.93. | PASS: confidence 86.9, similarity 0.963 |
| `sample_mixed_digital_scanned.pdf` | Page 1 is a digital text page (committee minutes), page 2 is the scanned page above. | Page 1 uses the text layer with no OCR (`ocr_text` is null); page 2 uses OCR with an empty `raw_text`; two pages, correct page indices, no flags or warnings. | PASS |

Observed errors, so the limits are visible:

- **Clean scan:** six small errors from specks and look-alike characters: a stray `.` after the date, a stray `.` before "the caretaker", a stray `-` in the block C table row, a stray `"` before "none", `C` read as `c`, and `CC0` read as `CCO`. All dates, times and names are correct.
- **Degraded scan:** the body text is intact, but "Mon 12 Oct" became `120ct`, the table row for block A gained junk characters, "Secretary" became `Sectetary`, and the small footer line is mostly unreadable. This is why the degraded test checks fixed key facts and a looser similarity instead of exact text.
- **All scans:** table columns are separated by single spaces, so the column layout is lost (one line per row is kept). "Page 1 of 1" shares a line with the footer note, so `printed_page_label` stays `null`.

How the fixtures were made: the clean page is the supplied specimen. The degraded page was rendered at 200 DPI, rotated 2 degrees, resized to 60%, blurred (Gaussian radius 1), given lower contrast (x0.85, 12 levels darker) and Gaussian noise (sigma 7, seed 42), then saved as JPEG quality 35 and embedded in a PDF of the same page size. The mixed PDF is a Helvetica text page made with reportlab, merged with the clean scan using pypdf. The manual transcription in the test file was typed from the page image.

This is one document with mild degradation, so it shows that the pipeline works end to end and handles mixed routing, but it says little about heavily degraded scans or phone photos. Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Tests

```bash
python -m unittest discover -s tests -v      # or: pytest -q (pip install pytest)
```

Run from the repository root. `pypdfium2`, `Pillow` and `numpy` are required to run the tests at all; the rest is optional and the affected tests skip automatically:

- no `reportlab`: the validation and pipeline test classes skip (they build their PDFs with it);
- no `pypdf`: only the encrypted-PDF test skips;
- no Tesseract (or no `pytesseract`): the two synthetic OCR scan tests and the nine OCR sample tests skip (11 in total). The OCR fallback, OCR flag and image-detection tests stub OCR, so they still run.

The PDF Processing suite (`tests/test_pdf_processing.py`) has 49 tests: validation, routing, cleaning, preprocessing, the pipeline contract, OCR fallback and flags (`ocr_unavailable`, `ocr_failed`, `low_ocr_confidence`, `ocr_empty_used_text_layer`, `image_detection_failed`, `non_english_script`), and the three OCR sample PDFs described above.

# Chunking

Splits the page-level JSON from `process_pdf` into chunk-level JSON for embedding and retrieval. Method: **page-bounded recursive chunking**. Standard library only, CPU only, no external APIs. No embedding model or tokenizer is assumed.

| Step | Method |
|---|---|
| Page boundary | Every PDF page is chunked on its own, so a chunk never contains text from two pages |
| Splitting | Recursive: paragraphs, then lines, then sentences, then words; a single unbroken run longer than the chunk size is cut by characters as a last resort |
| Packing | Neighbouring pieces are merged up to `chunk_size`; the start of each chunk repeats up to `chunk_overlap` of the end of the previous chunk on the same page |
| Input text | `page["text"]` (the cleaned text); `raw_text`, `ocr_text` and the input pages are never modified |

## Usage

```python
import pdf_processing as pp
import chunking as ck

result = pp.process_pdf("lecture1.pdf")
chunked = ck.chunk_document(result)                       # provisional defaults

cfg = ck.ChunkingConfig(chunk_size=800, chunk_overlap=100)
chunked = ck.chunk_document(result, cfg)

for chunk in chunked["chunks"]:
    chunk["text"]                 # text to embed and to show as evidence
    chunk["pdf_page_index"]       # 1-based, use for citations
    chunk["printed_page_label"]   # printed page number or None

ck.save_chunks_json(chunked, "lecture1.chunks.json")
chunked = ck.load_chunks_json("lecture1.chunks.json")
```

`chunk_document` also accepts a result loaded with `pp.load_result_json`. Two lower-level functions are available: `chunk_pages(pages, cfg)` for a list of page dicts, and `split_text(text, cfg)`, which returns `(char_start, char_end)` spans for one page's text.

## Settings

All settings live in `ChunkingConfig` and are validated (`chunk_size >= 1`, `0 <= chunk_overlap < chunk_size`).

| Setting | Default | Meaning |
|---|---|---|
| `chunk_size` | 1000 | Maximum chunk length, in `length_unit` (provisional) |
| `chunk_overlap` | 150 | Target overlap between neighbouring chunks, in `length_unit` (provisional) |
| `length_unit` | `"characters"` | Label stored in the output; change it only together with `length_function` |

The defaults are provisional starting values (1000 characters with 15% overlap), not tuned results. Sizes are measured in characters until the embedding model and tokenizer are confirmed. Final values will be selected after the embedding model and its tokenizer are confirmed.

**Switching to tokens later.** Pass the tokenizer's length function and set the sizes in tokens. Nothing else changes:

```python
count = lambda s: len(tokenizer.encode(s))       # whichever tokenizer is selected
cfg = ck.ChunkingConfig(chunk_size=256, chunk_overlap=40, length_unit="tokens")
chunked = ck.chunk_document(result, cfg, length_function=count)
```

## Output

`chunk_document` returns `{doc_id, source_file, page_count, config, chunks, warnings}`. Each chunk:

```text
chunk_id             "<doc_id>_p<pdf_page_index>_c<chunk_index>", unique within a document
doc_id
chunk_index          0-based position over the whole document, in page order
pdf_page_index       1-based PDF page (use for citations)
printed_page_label   printed page number from pdf_processing, or null
page_chunk_index     0-based position within its page (restarts on every page)
text                 the chunk text, trimmed; always equals page text[char_start:char_end]
char_start, char_end offsets into that page's cleaned "text" (end exclusive)
length               length of text in length_unit
source_type          "text_layer" or "ocr": where the page text came from
flags                the page's quality flags (for example "low_ocr_confidence")
```

`warnings` lists pages that produced no chunks (for example `Page 2 has no text; no chunks created.`), so an empty or failed scan is never silently lost. If a page's cleaned text is empty, check its `quality.flags` in the page-level JSON.

## How chunks are made

1. A page whose text fits in `chunk_size` becomes one chunk.
2. Otherwise the text is cut after each paragraph break; pieces that still do not fit are cut at line breaks, then sentence ends (`.`, `!`, `?`), then spaces, and finally by characters.
3. Small neighbouring pieces are merged up to `chunk_size`. A piece that is too big is split by itself and is never glued to the pieces on either side, so a split paragraph does not run into the next paragraph.
4. Each chunk after the first starts with trailing pieces of the previous chunk whose total length is at most `chunk_overlap` (and only if the new chunk still fits).

Limits:

- Overlap is made of whole pieces, so it can be smaller than `chunk_overlap`, and is zero when the last piece of a chunk is longer than `chunk_overlap`. There is never overlap across a page boundary.
- The sentence rule is a simple English pattern; an abbreviation such as "Fig. 3" can be treated as a sentence end. Tables are kept line by line, as in the page JSON.
- Chunk text is not lowercased. For BM25, build the search copy from the chunk text, for example `pp.make_search_text(chunk["text"])`.
- Chunking does not repair OCR errors; chunks from OCR pages carry `source_type: "ocr"` and the page flags so they can be filtered or weighted downstream.

## Chunking tests

`tests/test_chunking.py` (43 tests) covers config validation, each splitting level, size and overlap limits on generated text (no text is lost, offsets are exact, every chunk fits), custom length functions, page boundaries, preserved metadata, chunk indexes, empty pages, input immutability and JSON round trip. It also chunks real `process_pdf` output: a generated digital PDF with printed page numbers, a PDF with a blank page, and `sample_mixed_digital_scanned.pdf` (text-layer page plus scanned page).

```bash
python -m unittest discover -s tests -v      # runs the PDF Processing and Chunking tests together
```

The core tests (39) need only the standard library. The four tests that run the real pipeline skip when `reportlab` is missing, and the scanned-page test also skips when Tesseract is missing.
