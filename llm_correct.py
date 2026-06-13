"""Constrained LLM cleanup for OCR residue the n-gram LM can't touch.

The diacritic LM (`vndiacritic.py`) only changes a syllable's tone marks. This
adds a *constrained* LLM pass for the leftovers — merged tokens (`mộtstồn`),
stray symbols (`ta‹là`), and trailing garbage. A local Vietnamese instruct model
proposes a rewrite, but we accept its edits ONLY where they don't touch a valid
Vietnamese word, so it can never change meaning (`kiếm`->`súng`) or hallucinate.

Guards (why an unconstrained LLM is unsafe for subtitles — proven on real data):
  * valid Vietnamese words are protected and reverted if the LLM alters them;
  * inserted (invented) words are dropped;
  * a non-word may only be replaced by a high-character-overlap fix (a split or
    typo repair), never a low-overlap guess;
  * lines with no valid word at all are left as-is (no anchor to trust);
  * any CJK / out-of-charset output is rejected.

Runs on GPU (fp16) if available — ~1.5s/line, and lines without garbage are
skipped entirely. Model + HF cache live on G: (F: is full).

Usage:
    .venv\\Scripts\\python.exe llm_correct.py            # -> results/pytesseract_llm (preview)
    .venv\\Scripts\\python.exe llm_correct.py --inplace  # overwrite results/pytesseract
"""
import os
os.environ.setdefault("HF_HOME", r"G:\AI\models\hf")
import re
import io
import glob
import difflib
import argparse

MODEL_DIR = r"G:\AI\models\Qwen2.5-3B-Instruct"
VOCAB_MODEL = "vnlm/models/vi.pkl.gz"
SRC_DIR = "./results/pytesseract"

WORD = re.compile(r"[A-Za-zÀ-ỹ]+")
CJK = re.compile(r"[　-鿿＀-￯]")

SYS = (
    "Sửa lỗi OCR phụ đề tiếng Việt. Chỉ trả MỘT dòng kết quả, không giải thích.\n\n"
    "ƯU TIÊN XỬ LÝ (theo thứ tự):\n"
    "1. TÁCH CHỮ DÍNH: tách từ bị dính do OCR.\n"
    "   vd: 'tôibiết' → 'tôi biết' | 'anhấy' → 'anh ấy'\n"
    "2. SỬA KÝ TỰ/DẤU THANH: sửa OCR đọc sai chữ cái hoặc dấu.\n"
    "   vd: '1à' → 'là' | '0ng' → 'ông' | 'hoi' → 'hỏi'\n"
    "3. XOÁ RÁC: xoá ký tự/cụm không phải tiếng Việt, vô nghĩa trong câu.\n"
    "   vd: '§§' '|||' '£¥' chuỗi ký hiệu lạ\n\n"
    "RÀNG BUỘC:\n"
    "- GIỮ NGUYÊN chữ hoa/thường và mọi dấu câu (kể cả '...' '-' '!').\n"
    "- KHÔNG thêm từ, KHÔNG bịa, KHÔNG đoán từ thiếu.\n"
    "- Toàn bộ là rác → trả về chuỗi rỗng.\n"
    "- Câu đúng rồi → in lại y nguyên.\n\n"
    "VÍ DỤ:\n"
    "Input:  tôikhông biết§§ làm sao\n"
    "Output: tôi không biết làm sao\n\n"
    "Input:  Anhấy đã 1àm điều đó r0i.\n"
    "Output: Anh ấy đã làm điều đó rồi.\n\n"
    "Input:  Cô ấy nói... không sao.\n"
    "Output: Cô ấy nói... không sao.\n\n"
    "Input:  ||| §£¥ @@ ~~~\n"
    "Output: \n\n"
    "Input:  Tôi yêu em, dùdù thế nào.\n"
    "Output: Tôi yêu em, dù thế nào."
)
# SHOTS = [
#     ("Có chuyện gì thế? S", "Có chuyện gì thế?"),
#     ("Người đời tôn trọng và kính sợ", "Người đời tôn trọng và kính sợ"),
#     ("danh tiếng đã vang khắp thế giối...", "danh tiếng đã vang khắp thế giới..."),
#     ("Qa Cái gì thế này 2008", "Cái gì thế này"),
# ]


class LLMCorrector:
    def __init__(self, model_dir=MODEL_DIR, vocab_model=VOCAB_MODEL):
        import truststore; truststore.inject_into_ssl()
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from vndiacritic import DiacriticModel
        self.torch = torch
        self.vocab = set(DiacriticModel.load(vocab_model).uni)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        cfg = AutoConfig.from_pretrained(model_dir)
        # Multimodal models (e.g. Gemma 4 / 3n) need a processor + the image-text
        # class even for text-only use; plain LLMs (Qwen) use tokenizer + CausalLM.
        self.multimodal = hasattr(cfg, "vision_config")
        if self.multimodal:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.proc = AutoProcessor.from_pretrained(model_dir)
            self.tok = self.proc.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_dir, dtype=dtype).to(self.dev).eval()
        else:
            from transformers import AutoModelForCausalLM
            self.proc = None
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=dtype).to(self.dev).eval()

    def _isword(self, tok):
        core = tok.strip(".,!?;:'\"()-–…").lower()
        return bool(core) and core in self.vocab

    def suspect(self, line):
        """True if a line has a letter-token that isn't a valid Vietnamese word."""
        return any(not self._isword(w) for w in WORD.findall(line))

    def _llm_raw(self, line):
        c = (lambda t: [{"type": "text", "text": t}]) if self.multimodal else (lambda t: t)
        msgs = [{"role": "system", "content": c(SYS)}]
        # for u, a in SHOTS:
        #     msgs += [{"role": "user", "content": c(u)}, {"role": "assistant", "content": c(a)}]
        msgs.append({"role": "user", "content": c(line)})
        if self.multimodal:
            inp = self.proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self.dev)
        else:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = self.tok(text, return_tensors="pt").to(self.dev)
        in_len = inp["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=len(line) + 16, do_sample=False)
        r = self.tok.decode(out[0][in_len:], skip_special_tokens=True).strip()
        r = r.splitlines()[0] if r else r
        return re.sub(r"\s*\([^)]*\)\s*$", "", r.strip("\"'")).strip()

    def _constrain(self, orig, out):
        O, L = orig.split(), out.split()
        if not any(self._isword(t) for t in O):
            return orig
        res = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, O, L).get_opcodes():
            if tag == "equal":
                res += O[i1:i2]
            elif tag == "delete":
                res += [t for t in O[i1:i2] if self._isword(t)]
            elif tag == "insert":
                pass
            elif tag == "replace":
                o_chunk, l_chunk = O[i1:i2], L[j1:j2]
                sim = difflib.SequenceMatcher(
                    None, "".join(o_chunk).lower(), "".join(l_chunk).lower()).ratio()
                if any(self._isword(t) for t in o_chunk) or sim < 0.6:
                    res += o_chunk
                else:
                    res += [t for t in l_chunk if not CJK.search(t)]
        return " ".join(res)

    def correct_line(self, line):
        if not line.strip() or not self.suspect(line):
            return line
        raw = self._llm_raw(line)
        return self._constrain(line, raw) if raw else line

    def correct_text(self, text):
        return "\n".join(self.correct_line(ln) for ln in text.splitlines())


def main():
    p = argparse.ArgumentParser(description="Constrained LLM OCR cleanup.")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--model", default=MODEL_DIR)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    corr = LLMCorrector(args.model)
    out_dir = SRC_DIR if args.inplace else "./results/pytesseract_llm"
    os.makedirs(out_dir, exist_ok=True)
    changed = 0
    for path in sorted(glob.glob(f"{SRC_DIR}/*.txt")):
        text = io.open(path, encoding="utf-8").read()
        fixed = corr.correct_text(text)
        if fixed != text:
            changed += 1
            if args.verbose:
                for a, b in zip(text.split("\n"), fixed.split("\n")):
                    if a != b:
                        print(f"- {a!r}\n+ {b!r}")
        with io.open(os.path.join(out_dir, os.path.basename(path)), "w", encoding="utf-8") as f:
            f.write(fixed)
    print(f"\nLLM-corrected {changed} file(s) -> {out_dir}")


if __name__ == "__main__":
    main()
