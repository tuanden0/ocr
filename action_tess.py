import os
import re
import difflib
import argparse
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pytesseract

# Define configuration parameters
IN_DIR = "./dataset/processed"
OUT_DIR = "./results/pytesseract"
LANG = "vie"
# --oem 1 -> LSTM engine (best for Vietnamese), --psm 6 -> assume a uniform
# block of text (subtitles can span one OR two lines, so 6 beats single-line 7).
CONFIG = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
# Per-block config: each segmented block is a single text line.
LINE_CONFIG = "--oem 1 --psm 7 -c preserve_interword_spaces=1"
# Target text-line height in pixels. The LSTM model was trained on ~30-50px
# x-heights; the source frames render at ~160px, which loses dot-below tone
# marks. Down-scaling each frame so a line is ~this tall fixes them.
TARGET_LINE_HEIGHT = 70

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# A character we are willing to keep in the final output.
LETTER = re.compile(r"[A-Za-zÀ-ỹ]")
# Pure numeric / punctuation token (e.g. "3.." or "7", subtitle indices).
NUMERIC = re.compile(r"^[0-9.,:;!?%()\-–…'\"]+$")
# Symbols and CJK ranges that only show up as OCR garbage from the non-Vietnamese
# (Japanese) on-screen credits baked into some frames.
BAD = re.compile(r"[#%*&@/\\<>~^°|　-鿿＀-￯]")


def get_files():
    files = []
    for f in os.listdir(IN_DIR):
        img = os.path.join(IN_DIR, f)
        if os.path.isfile(img) and f.lower().endswith(IMG_EXTS):
            p, _ = os.path.splitext(f)
            txt = os.path.join(OUT_DIR, f"{p}.txt")
            files.append({"img": img, "txt": txt, "name": p})
    return files


def _line_height(gray):
    """Estimate the median text-line height (in px) from the ink-row profile."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    has_ink = (bw < 128).any(axis=1)
    bands, start = [], None
    for i, ink in enumerate(has_ink):
        if ink and start is None:
            start = i
        elif not ink and start is not None:
            bands.append(i - start)
            start = None
    if start is not None:
        bands.append(len(has_ink) - start)
    bands = [b for b in bands if b > 10]  # ignore stray speckles
    return float(np.median(bands)) if bands else float(gray.shape[0])


def _rescale(img):
    """Down-/up-scale grayscale so a text line is ~TARGET_LINE_HEIGHT and border it.

    The frames render text far larger than the LSTM model expects, which loses
    dot-below tone marks. Grayscale (not hard-binarized) is fed to Tesseract so
    anti-aliased diacritic dots survive and the engine thresholds internally.
    """
    scale = TARGET_LINE_HEIGHT / _line_height(img)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)


def preprocess(path):
    """Whole-frame grayscale rescale (used by the non-segmented path and tools)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return _rescale(img)


def _runs(present, min_run, min_gap):
    """Runs of True in a boolean profile, merging gaps smaller than min_gap."""
    runs, start = [], None
    for i, v in enumerate(present):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(present)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(s, e) for s, e in merged if e - s >= min_run]


def _blocks(gray, pad=12):
    """Split a frame into text blocks: rows, then columns separated by wide gaps.

    A gap wider than ~2.5 line heights separates the Vietnamese subtitle from the
    baked-in Japanese on-screen credits, so they get OCR'd apart and the credit
    block's garble no longer corrupts the subtitle line.
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = bw < 128
    h, w = gray.shape
    for y0, y1 in _runs(ink.any(axis=1), min_run=12, min_gap=10):
        lh = y1 - y0
        for x0, x1 in _runs(ink[y0:y1].any(axis=0),
                            min_run=max(4, int(lh * 0.3)), min_gap=int(lh * 2.5)):
            yield gray[max(0, y0 - pad):min(h, y1 + pad),
                       max(0, x0 - pad):min(w, x1 + pad)]


def _keep(token):
    if BAD.search(token):
        return False
    # Judge the token by its core, ignoring leading/trailing punctuation such as
    # the "..." in "họ..." that would otherwise drag the letter ratio down.
    core = token.strip(".,:;!?%()-–…'\"")
    if not core:
        return False
    if NUMERIC.match(core):
        return True
    # A real Vietnamese word never mixes letters and digits inside one token, so
    # letter+digit blobs (e.g. "ER7n7", "S4") are credit-OCR garble -> drop.
    if any(c.isdigit() for c in core) and any(LETTER.match(c) for c in core):
        return False
    letters = sum(1 for c in core if LETTER.match(c))
    return letters > 0 and letters / len(core) >= 0.5


def clean(text):
    """Normalize diacritics to NFC and drop non-Vietnamese garbage tokens."""
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.splitlines():
        tokens = [w for w in line.split() if _keep(w)]
        if tokens:
            lines.append(re.sub(r"[ \t]+", " ", " ".join(tokens)).strip())
    return "\n".join(lines)


_EDGE_PUNCT = ".,!?;:'\"()-–…"


def _trim_edges(text, vocab):
    """Drop credit-OCR garble at line edges (e.g. the '©海空'->'Qa' and '委員会'
    ->'2008' around "Cái gì thế này"). A leading/trailing token is dropped only if
    it is NOT a valid Vietnamese word AND is either very short (<=2 chars, so real
    names like 'Ringo' survive) or a >=3-digit number (so 'cấp 3' survives) — and
    only while >=2 real Vietnamese words remain on the line."""
    def is_word(tok):
        core = tok.strip(_EDGE_PUNCT).lower()
        return bool(core) and core in vocab

    def is_junk(tok):
        core = tok.strip(_EDGE_PUNCT)
        if not core or is_word(tok):
            return False
        if core.isdigit():
            return len(core) >= 3   # drop years like '2008', keep 'cấp 3', 'chương 13'
        return len(core) <= 2       # drop short non-word letters like 'Qa', stray 'X'

    out = []
    for line in text.splitlines():
        toks = line.split()
        if sum(is_word(t) for t in toks) >= 2:
            while len(toks) > 1 and is_junk(toks[0]):
                toks.pop(0)
            while len(toks) > 1 and is_junk(toks[-1]):
                toks.pop()
        out.append(" ".join(toks))
    return "\n".join(out)


def ocr(path, segment=False):
    """OCR a frame. With segmentation (opt-in), each text block is rescaled and
    read on its own (psm 7); otherwise the whole frame is read as one block
    (psm 6, the proven default). Segmentation only helps when on-screen credits
    are separated by a wide gap; otherwise psm 7 per block can regress."""
    if not segment:
        raw = pytesseract.image_to_string(preprocess(path), lang=LANG, config=CONFIG)
        return clean(raw)

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"could not read image: {path}")
    lines = []
    for crop in (list(_blocks(gray)) or [gray]):
        raw = pytesseract.image_to_string(_rescale(crop), lang=LANG, config=LINE_CONFIG)
        text = clean(raw)
        if text:
            lines.append(text)
    return "\n".join(lines)


def _ocr_data(path):
    """Tesseract words with per-word confidence and a (block, par, line) key so
    multi-line frames keep their line breaks."""
    d = pytesseract.image_to_data(preprocess(path), lang=LANG, config=CONFIG,
                                  output_type=pytesseract.Output.DICT)
    out = []
    for t, c, b, p, l in zip(d["text"], d["conf"], d["block_num"],
                             d["par_num"], d["line_num"]):
        if t.strip() and str(c).lstrip("-").isdigit() and int(c) >= 0:
            out.append((t, int(c), (b, p, l)))
    return out


def _data_text(wc):
    """Reconstruct multi-line text from _ocr_data output (line breaks preserved)."""
    lines, order = {}, []
    for w, _, key in wc:
        if key not in lines:
            lines[key], _ = [], order.append(key)
        lines[key].append(w)
    return clean("\n".join(" ".join(lines[k]) for k in order))


def fuse(tess_words, tess_confs, viet_words, vocab, conf_thr=60):
    """Word-level Tesseract+VietOCR merge: keep Tesseract's structure, but where a
    Tesseract word is low-confidence, take VietOCR's aligned word if it's a valid
    Vietnamese word (or Tesseract's wasn't). Fixes partial misreads (`éhiếc`->
    `chiếc`) that whole-line picking misses, without losing confident words."""
    def isw(w):
        return w.strip(_EDGE_PUNCT).lower() in vocab

    sm = difflib.SequenceMatcher(None, [w.lower() for w in tess_words],
                                 [w.lower() for w in viet_words])
    res = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "delete"):           # agree, or Tesseract-only -> keep
            res += tess_words[i1:i2]
        elif tag == "insert":                     # VietOCR-only -> add only real words
            res += [w for w in viet_words[j1:j2] if isw(w)]
        elif tag == "replace":
            t, c, v = tess_words[i1:i2], tess_confs[i1:i2], viet_words[j1:j2]
            if len(t) == len(v):
                for k in range(len(t)):
                    take_v = c[k] < conf_thr and (isw(v[k]) or not isw(t[k]))
                    res.append(v[k] if take_v else t[k])
            else:  # chunk lengths differ: take VietOCR only if Tesseract is weak here
                res += v if (sum(c) / len(c) < conf_thr and any(isw(w) for w in v)) else t
    return " ".join(res)


def process(file):
    """OCR a single image (writing/ensemble/correction happen in the caller so the
    non-picklable VietOCR + LM models stay in the main process). With ensemble we
    return per-word data so the caller can fuse with VietOCR."""
    img = file["img"]
    if file.get("need_conf"):
        wc = _ocr_data(img)
        return img, file["txt"], _data_text(wc), wc
    return img, file["txt"], ocr(img, file.get("segment", False)), None


def main():
    parser = argparse.ArgumentParser(description="Vietnamese OCR with Tesseract.")
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count(),
                        help="number of parallel worker processes")
    parser.add_argument("--correct", action="store_true",
                        help="restore Vietnamese diacritics with the trained LM")
    parser.add_argument("--model", default="vnlm/models/vi.pkl.gz",
                        help="diacritic LM model for --correct")
    parser.add_argument("--keep-boost", type=float, default=14.0, dest="keep_boost",
                        help="--correct: bias toward the OCR reading (higher = safer)")
    parser.add_argument("--segment", action="store_true",
                        help="OCR each text block separately (opt-in; see ocr())")
    parser.add_argument("--ensemble", action="store_true",
                        help="fall back to VietOCR on low-confidence frames")
    parser.add_argument("--conf-threshold", type=float, default=70.0, dest="conf_threshold",
                        help="--ensemble: fuse with VietOCR when mean confidence < this")
    parser.add_argument("--word-conf-threshold", type=float, default=40.0, dest="word_conf_threshold",
                        help="--ensemble: also fuse when any single word's conf < this")
    parser.add_argument("--llm-correct", action="store_true", dest="llm_correct",
                        help="constrained LLM cleanup of merges/garble (GPU; see llm_correct.py)")
    parser.add_argument("--llm-model", default=None, dest="llm_model",
                        help="local dir for the LLM (default: llm_correct.MODEL_DIR)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    files = get_files()
    if not files:
        print("No files found in the input directory.")
        return
    for f in files:
        f["segment"] = args.segment
        f["need_conf"] = args.ensemble

    corrector = None
    if args.correct:
        from vndiacritic import DiacriticModel
        corrector = DiacriticModel.load(args.model)
        corrector.keep_boost = args.keep_boost
        print(f"Diacritic correction on (model {args.model}, keep_boost {args.keep_boost}).")

    predictor = vietocr_ocr = vocab = trim_vocab = None
    if args.correct or args.ensemble:
        from vndiacritic import DiacriticModel
        uni = (corrector or DiacriticModel.load(args.model)).uni
        vocab = set(uni)                                       # broad: engine scoring
        trim_vocab = {w for w, c in uni.items() if c >= 100}   # strict: edge garbage
    if args.ensemble:
        import action_vietocr
        predictor, vietocr_ocr = action_vietocr.load_predictor(), action_vietocr.ocr
        print(f"Ensemble on: word-level VietOCR fusion when mean conf < "
              f"{args.conf_threshold} or any word < {args.word_conf_threshold}.")

    llm = None
    if args.llm_correct:
        from llm_correct import LLMCorrector, MODEL_DIR
        llm = LLMCorrector(args.llm_model or MODEL_DIR)
        print(f"LLM cleanup on (model {args.llm_model or MODEL_DIR}, device {llm.dev}).")

    def vn_score(t):  # count of word-tokens that are known Vietnamese syllables
        toks = re.findall(r"[A-Za-zÀ-ỹ]+", t.lower())
        return sum(1 for w in toks if w in vocab)

    n_fused = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, f): f for f in files}
        for future in as_completed(futures):
            img = futures[future]["img"]
            try:
                _, txt_path, text, wc = future.result()
                if predictor and wc:
                    confs = [c for _, c, _ in wc]
                    weak = confs and (sum(confs) / len(confs) < args.conf_threshold
                                      or min(confs) < args.word_conf_threshold)
                    if weak:
                        vt = vietocr_ocr(predictor, img)
                        fused = clean(fuse([w for w, _, _ in wc], confs, vt.split(), trim_vocab)) if vt else ""
                        if fused and vn_score(fused) >= vn_score(text):
                            text, n_fused = fused, n_fused + 1
                if not text:
                    if os.path.exists(txt_path):  # drop stale output from a prior run
                        os.remove(txt_path)
                    print(f"Skipped (empty): {img}")
                    continue
                if corrector:
                    text = corrector.restore_text(text)
                if llm:
                    text = llm.correct_text(text)
                if trim_vocab:
                    text = _trim_edges(text, trim_vocab)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Processed: {img}")
            except Exception as e:
                print(f"Failed: {img} ({e})")
    if args.ensemble:
        print(f"VietOCR word-fusion applied on {n_fused} frame(s).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted.")
