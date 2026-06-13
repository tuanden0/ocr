"""Vietnamese OCR with VietOCR (pretrained transformer recognizer).

VietOCR is a *line recognizer*, which fits the pre-cropped subtitle strips well.
The frames, though, are very wide and may hold two stacked lines plus baked-in
Japanese credits off to the side, so we segment each frame into text blocks first
(by gaps in the ink profile) and recognise each block on its own. Garbage from the
non-Vietnamese credits is then dropped by the shared `action_tess.clean` filter.

Model files are kept locally under vnlm/vietocr/ (downloaded out-of-band because
this machine's cert store breaks VietOCR's own requests-based downloader):
    base.yml, vgg-transformer.yml, vgg_transformer.pth

Usage:
    .venv\\Scripts\\python.exe action_vietocr.py            # -> results/vietocr/*.txt
    .venv\\Scripts\\python.exe action_vietocr.py --correct  # + diacritic LM (optional)
"""
import os
import io
import argparse

import cv2
import yaml
import numpy as np
from PIL import Image

from action_tess import clean

IN_DIR = "./dataset/processed"
OUT_DIR = "./results/vietocr"
VIETOCR_DIR = "vnlm/vietocr"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_predictor():
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    cfg = {}
    for name in ("base.yml", "vgg-transformer.yml"):
        with io.open(os.path.join(VIETOCR_DIR, name), encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f))
    cfg = Cfg(cfg)
    cfg["weights"] = os.path.join(VIETOCR_DIR, "vgg_transformer.pth")
    cfg["device"] = "cpu"
    cfg["predictor"]["beamsearch"] = False
    return Predictor(cfg)


def _bands(mask, axis, min_run, min_gap):
    """Yield (start, end) runs of ink along `axis`, merging gaps < min_gap."""
    present = mask.any(axis=axis)
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


def segment(path, pad=10):
    """Split a frame into text-block crops: rows first, then wide column gaps."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"could not read image: {path}")
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = bw < 128
    H, W = gray.shape

    crops = []
    for y0, y1 in _bands(ink, axis=1, min_run=12, min_gap=10):
        lh = y1 - y0
        # Split the row into blocks separated by gaps wider than ~2.5 line heights
        # (isolates the Vietnamese subtitle from side credits, but never words).
        # min_run drops slivers of noise (single stray marks).
        for x0, x1 in _bands(ink[y0:y1], axis=0, min_run=int(lh * 0.4),
                             min_gap=int(lh * 2.5)):
            yy0, yy1 = max(0, y0 - pad), min(H, y1 + pad)
            xx0, xx1 = max(0, x0 - pad), min(W, x1 + pad)
            crops.append(gray[yy0:yy1, xx0:xx1])
    return crops or [gray]


def ocr(predictor, path):
    lines = []
    for crop in segment(path):
        pil = Image.fromarray(crop).convert("RGB")
        text = predictor.predict(pil).strip()
        if text:
            lines.append(text)
    return clean("\n".join(lines))


def get_files():
    files = []
    for f in os.listdir(IN_DIR):
        img = os.path.join(IN_DIR, f)
        if os.path.isfile(img) and f.lower().endswith(IMG_EXTS):
            p, _ = os.path.splitext(f)
            files.append((img, os.path.join(OUT_DIR, f"{p}.txt")))
    return files


def main():
    parser = argparse.ArgumentParser(description="Vietnamese OCR with VietOCR.")
    parser.add_argument("--correct", action="store_true",
                        help="restore diacritics with the trained LM afterwards")
    parser.add_argument("--model", default="vnlm/models/vi.pkl.gz")
    parser.add_argument("--keep-boost", type=float, default=14.0, dest="keep_boost")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    files = get_files()
    if not files:
        print("No files found in the input directory.")
        return

    print("Loading VietOCR model...")
    predictor = load_predictor()

    corrector = None
    if args.correct:
        from vndiacritic import DiacriticModel
        corrector = DiacriticModel.load(args.model)
        corrector.keep_boost = args.keep_boost

    for i, (img, txt) in enumerate(files, 1):
        try:
            text = ocr(predictor, img)
            if not text:
                print(f"[{i}/{len(files)}] Skipped (empty): {img}")
                continue
            if corrector:
                text = corrector.restore_text(text)
            with io.open(txt, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[{i}/{len(files)}] {img}")
        except Exception as e:
            print(f"[{i}/{len(files)}] Failed: {img} ({e})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted.")
