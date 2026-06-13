# CLAUDE.md

Guidance for working in this repo. For deeper rationale (why the design is the way
it is), see the project memory `vietnamese-ocr-pipeline`.

## What this project does

OCRs Vietnamese subtitle strips extracted from video. The images in
`dataset/processed/` are wide (~5120px), short (~250–450px tall), black text on a
transparent/white background, sometimes with baked-in Japanese on-screen credits.
The goal is accurate Vietnamese text output in `results/`.

**`action_tess.py` (Tesseract) is the maintained backend** and the one with all the
accuracy work below; `action_vietocr.py` (VietOCR) is the alternative/ensemble
backend. (An earlier `action_ezocr.py` EasyOCR backend was removed.)

## Environment

- **Python 3.12**, managed with `uv` (see `pyproject.toml`, `uv.lock`).
- **Always run via the venv interpreter:** `.venv\Scripts\python.exe` on Windows.
  The Bash tool's default Python is a *different* interpreter and lacks the
  installed packages (`torch`, `wordfreq`, etc.).
- **Tesseract** must be installed and on PATH: `F:\Program Files\Tesseract-OCR\tesseract.exe`.
  The `vie` traineddata is the `tessdata_best` model (already best available).
- Install deps: `uv sync --system-certs` (TLS on this machine needs `--system-certs`).
  `pyproject.toml` now pins **everything** incl. `vietocr`, `transformers`, and the
  **CUDA torch** (`torch==2.11.0+cu128` via the `pytorch-cu128` index in `[tool.uv.sources]`),
  so a plain `uv sync` no longer drops the GPU build or the LLM/VietOCR deps. The
  manual `uv pip install` lines in §4/§5 are now redundant (kept for reference).
  `vndiacritic.py` itself needs only the standard library.

## When to use what (quick guide)

| Your situation | Run this |
|----------------|----------|
| **Default: best Vietnamese text in one command** | `action_tess.py --correct` |
| Want raw OCR only (to A/B, debug, or re-correct later) | `action_tess.py` |
| Already have OCR output; correct it (preview first) | `vndiacritic.py correct` → `--inplace` to apply |
| Correction changed too much / broke good text | re-run with higher `--keep-boost` (e.g. 25), or preview before applying |
| Want VietOCR to auto-cover Tesseract's weak frames | `action_tess.py --correct --ensemble` (vocab-scored pick) |
| Manually compare engines on a frame | `action_vietocr.py`, then diff `results/vietocr` vs `results/pytesseract` |
| Residual merges/garble after `--correct` (and a GPU is free) | `llm_correct.py` (constrained; preview first) |
| You have more/better Vietnamese **subtitle** text | `vndiacritic.py train --grow` to improve the model |
| Tempted to swap Tesseract's dictionary to fix errors | **don't** — inert for the LSTM engine (see Gotchas) |
| Need to push OCR accuracy further than the above | `lstmtraining` fine-tune (large effort; not set up yet) |

Rule of thumb: **`action_tess.py --correct` is the everyday command.** Reach for
VietOCR only to compare, and retrain the LM only when you have new in-domain text.

## How to use

### 1. OCR (basic)  — *when: you want raw OCR, no correction*
```powershell
.venv\Scripts\python.exe action_tess.py            # OCR all images -> results/pytesseract/*.txt
.venv\Scripts\python.exe action_tess.py -j 8       # limit to 8 worker processes
```

### 2. OCR + Vietnamese diacritic correction (recommended)  — *when: normal day-to-day use*
```powershell
.venv\Scripts\python.exe action_tess.py --correct                  # one-pass OCR + LM correction
.venv\Scripts\python.exe action_tess.py --correct --keep-boost 25  # more conservative (fewer changes)
.venv\Scripts\python.exe action_tess.py --correct --ensemble       # + VietOCR fallback on weak frames
.venv\Scripts\python.exe action_tess.py --correct --llm-correct     # + constrained LLM cleanup (GPU)
```
`--correct` loads the trained LM (`vnlm/models/vi.pkl.gz`) and restores diacritics
after OCR. Higher `--keep-boost` = trusts the OCR more = fewer (safer) changes
(default 14). `--ensemble`: where a frame's mean confidence < `--conf-threshold` (70) OR any
single word's confidence < `--word-conf-threshold` (40), run VietOCR and **fuse
at the word level** (`fuse()`) — keep Tesseract's confident words and structure,
but swap a low-confidence Tesseract word for VietOCR's aligned word when that one
is a valid Vietnamese word (or Tesseract's wasn't). Fixes partial misreads
(`éhiếc`→`chiếc`) that the old whole-line pick missed, without losing the
confident parts (the leading `Ồ,`, the `!`). `--segment` (opt-in) OCRs text blocks separately; only helps when
on-screen credits are widely separated, so off by default. `--llm-correct` runs
the §5 constrained LLM cleanup as a final stage in the same command (needs the
GPU + model setup); the standalone `llm_correct.py` does the same on existing
results. Stage order: OCR → ensemble → diacritic LM → LLM cleanup.

### 3. Correct existing results separately (without re-OCR)  — *when: you already OCR'd and only want to (re)correct*
```powershell
.venv\Scripts\python.exe vndiacritic.py correct --verbose            # -> results/pytesseract_corrected (preview)
.venv\Scripts\python.exe vndiacritic.py correct --inplace            # overwrite results/pytesseract
```
Prefer the preview form when you want to diff before committing changes.

### 4. Alternative backend: VietOCR (pretrained transformer)  — *when: comparing engines or Tesseract struggles on a frame*
```powershell
.venv\Scripts\python.exe action_vietocr.py            # -> results/vietocr/*.txt
.venv\Scripts\python.exe action_vietocr.py --correct  # + diacritic LM
```
VietOCR is a pretrained Vietnamese line recognizer. Setup (one-time):
```powershell
uv pip install --system-certs vietocr
# VietOCR's own downloader fails on this machine's cert store, so fetch the model
# files manually into vnlm/vietocr/ (Invoke-WebRequest works with system certs):
#   https://vocr.vn/data/vietocr/config/base.yml
#   https://vocr.vn/data/vietocr/config/vgg-transformer.yml
#   https://vocr.vn/data/vietocr/vgg_transformer.pth   (~150MB)
```
The first run also pulls the torchvision `vgg19_bn` ImageNet backbone (~548MB) from
`download.pytorch.org` (that host's cert works) into the torch hub cache.
`action_vietocr.py` segments each frame into text blocks before recognising, so it
handles 1-/2-line frames. **Verdict so far:** on par with Tesseract+LM on clean
frames, occasionally better on tone marks, sometimes worse on messy frames with
overlapping text + Japanese. Tesseract+LM remains the default; VietOCR is here to
A/B and as a fallback.

### 5. Constrained LLM cleanup (optional, GPU)  — *when: residual merges/garble remain after `--correct`*
```powershell
.venv\Scripts\python.exe llm_correct.py            # -> results/pytesseract_llm (preview)
.venv\Scripts\python.exe llm_correct.py --inplace  # overwrite results/pytesseract
```
A local Vietnamese LLM (Qwen2.5-3B-Instruct) fixes the leftovers the n-gram LM
can't — merged tokens (`mộtstồn`→`một tồn`), stray symbols (`ta‹là`→`ta là`),
garbled non-words (`fihững`→`những`). **Heavily constrained** (`_constrain`): it
accepts an edit ONLY where the original token is not a valid Vietnamese word, so
it can never change meaning (proven: unconstrained it rewrote `kiếm`→`súng`,
`bỏ phiếu`→`bầu cử`). Lines without garbage are skipped (gating), so it's a
targeted pass, not a rewrite. One-time setup:
```powershell
uv pip install --system-certs transformers accelerate truststore
# CUDA torch (RTX 3060 present); download host's cert works:
uv pip install --system-certs --reinstall-package torch --reinstall-package torchvision `
    --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# model + HF cache on G: (F: is full); download elsewhere if slow:
$env:HF_HUB_ENABLE_HF_TRANSFER="1"; hf download Qwen/Qwen2.5-3B-Instruct --local-dir "G:\AI\models\Qwen2.5-3B-Instruct"
```
`llm_correct.py` sets `HF_HOME=G:\AI\models\hf` and uses GPU fp16 (~1.5s/line,
CPU fallback ~10-40s/line). `LLMCorrector` is **model-agnostic**: pass any local
dir via `--llm-model` (or `llm_correct.py --model`). It auto-detects multimodal
models (e.g. `gemma-4-E2B-it`, which has a `vision_config`) and loads them with
`AutoProcessor` + `AutoModelForImageTextToText`; plain LLMs (Qwen) use
`AutoTokenizer` + `AutoModelForCausalLM`. The same `_constrain` safety applies
regardless of model. `truststore` + `G:\AI\models\winca.pem` handle this
machine's cert store. Verdict: safe net-positive, but only ~12/465 frames change
— it's a niche cleanup, not a core stage.

### 6. Train / grow the diacritic model  — *when: you have new in-domain Vietnamese text*
```powershell
# fresh train (one sentence per line; .txt / .txt.gz / .tar.gz / a directory all work)
.venv\Scripts\python.exe vndiacritic.py train --corpus vnlm\corpus\vi.txt.gz --model vnlm\models\vi.pkl.gz

# grow an existing model with more in-domain text (counts accumulate)
.venv\Scripts\python.exe vndiacritic.py train --corpus more_text\ --model vnlm\models\vi.pkl.gz --grow
```
The trainer reads `.txt`, `.txt.gz`, `.tar.gz` (Leipzig), and **subtitle files
`.srt` / `.ass` / `.ssa`** — point `--corpus` at a folder and it **recurses into
all subfolders** (e.g. `Series/S01/ep.ass`), **skips `.git`/hidden entries**, and
**sniffs the BOM** for UTF-8/UTF-16. It strips timestamps, indices, and override
tags (`{\an8}`, `\N`, `<i>`, …). A ~16k-file anime library yields ~10M clean lines.

**Train on a folder of subtitles** (e.g. 20k `.srt`/`.ass` files):
```powershell
# preview that extraction looks clean before committing:
.venv\Scripts\python.exe -c "import vndiacritic as V, itertools; [print(repr(x)) for x in itertools.islice(V.iter_corpus_lines(r'D:\subs'), 30)]"

# grow the existing model with them (recommended — adds to the OpenSubtitles base):
.venv\Scripts\python.exe vndiacritic.py train --corpus "D:\subs" --model vnlm\models\vi.pkl.gz --grow
```
Caveats: subtitles must be **Vietnamese** and **UTF-8** (decoded as utf-8-sig,
undecodable bytes ignored — legacy TCVN3/VNI/Windows-1258 files will train on
garbage, so spot-check the preview). Bilingual subs add English noise.

**Domain matters:** train on conversational/subtitle text (OpenSubtitles vi works
well: `https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/vi.txt.gz`). A
news corpus is net-negative on this dialogue. More in-domain data = better.

## How the accuracy works (don't regress these)

`action_tess.py`:
- **Auto line-height scaling** (`preprocess()` + `TARGET_LINE_HEIGHT=70`): the source
  text is ~3–4x larger than Tesseract's LSTM sweet spot, which loses the dot-below
  tone marks. Downscaling each frame to a ~70px line height is the single biggest
  accuracy win. Feed **grayscale** (not hard-binarized) so anti-aliased diacritic
  dots survive.
- **Config:** `--oem 1 --psm 6` (LSTM, uniform block — handles 1- and 2-line frames).
- **`LANG = "vie"` only.** Do NOT use `vie+eng` (the English model strips Vietnamese
  diacritics) and never `jpn`. Garbage from Japanese credits is filtered in `clean()`.
- **Edge-trim** (`_trim_edges`, active with `--correct`/`--ensemble`): drops Japanese-
  credit garble at line ends (`Qa Cái gì thế này 2008` -> `Cái gì thế này`). Only
  removes a leading/trailing token that is NOT a frequent Vietnamese word (vocab with
  count >= 100, so stray junk like `qa` is caught but real words/names are not) AND is
  either a non-number <=2 chars (`Qa`, stray `X`) or a >=3-digit number (`2008`) — short
  numbers like `cấp 3` / `chương 13` are kept — and only while >=2 real words remain. A
  bare `S` can survive because `s` is a genuinely frequent token (English `It's`, romaji).

`vndiacritic.py` (diacritic LM):
- Trainable syllable n-gram + Viterbi restorer. Only ever changes a syllable's
  **diacritics**, never its letters (candidates share the diacritic-free skeleton).
- Guards: never corrects capitalized non-initial tokens or hyphenated honorifics
  (protects names like `Nhật`, `Ringo-chan`). `keep_boost` biases toward the OCR.
- Net positive (~2:1) but not perfect; residual errors (e.g. `kiêm`→`kiếm`) need
  trigram context — a possible future upgrade.

## Layout

| Path | What |
|------|------|
| `action_tess.py` | Tesseract OCR pipeline (maintained) + optional `--correct` |
| `vndiacritic.py` | Trainable Vietnamese diacritic-restoration LM (`train` / `correct`) |
| `llm_correct.py` | Optional constrained LLM cleanup (Qwen2.5-3B on GPU) for merges/garble |
| `action_vietocr.py` | VietOCR backend (pretrained transformer) + optional `--correct` |
| `train.py` | Unrelated: legacy Tesseract *retraining* flow. **Wipes `./data/`** — keep LM data in `vnlm/` |
| `dataset/raw`, `dataset/processed` | Source and pre-processed subtitle images |
| `results/pytesseract` | OCR text output (the maintained pipeline); `action_vietocr.py` writes `results/vietocr` when run |
| `vnlm/` | LM assets: `corpus/` (training text), `models/vi.pkl.gz` (+ `vi_opensubs.pkl.gz` baseline backup), `vietocr/` (VietOCR model files) |

## Gotchas

- `train.py` deletes everything in `./data/`; never store anything you care about there.
- The diacritic model object is not picklable (lambda defaultdicts), so `--correct`
  runs correction in the main process, not in OCR workers.
- Re-running `--correct` is idempotent on already-corrected output.
- **Don't bother with a custom Tesseract dictionary.** A `vie_sub.traineddata` was once
  built with an in-domain word DAWG, but the LSTM engine (`--oem 1`) ignores the
  dictionary even at `language_model_penalty_non_dict_word=1.0` — output is identical
  to stock `vie`. The real levers are preprocessing and the post-correction LM.
- **Don't fine-tune the LSTM for general use.** Stock `tessdata_best vie` is already
  font-independent; fine-tuning specializes to the training font and tends to regress
  other fonts. Synthetic subtitle text can't out-train Google's font variety.
