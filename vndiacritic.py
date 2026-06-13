"""Trainable Vietnamese diacritic-restoration language model.

A lightweight, fully local syllable n-gram LM with Viterbi decoding. It learns
real *contextual* probabilities from a Vietnamese corpus, so it knows that
"tình yêu" is far likelier than "tính yêu" even though "tính" is a commoner word
in isolation -- exactly the context the earlier dictionary approach lacked.

Design choices that keep it safe and useful for OCR post-correction:
  * It only ever changes a syllable's *diacritics*, never its letters: every
    candidate must share the OCR token's diacritic-free skeleton. So it cannot
    turn a word into an unrelated one (no "Dễ" -> "Bè").
  * A tunable "trust the OCR" bonus biases toward leaving the read text alone,
    so it overrides only when context clearly favours another diacritization.
  * It "grows": train again with --grow to accumulate counts from more text.

CLI:
    python vndiacritic.py train  --corpus vnlm/corpus --model vnlm/models/vi.pkl.gz
    python vndiacritic.py train  --corpus more_text/  --model vnlm/models/vi.pkl.gz --grow
    python vndiacritic.py correct --model vnlm/models/vi.pkl.gz            # -> *_corrected
    python vndiacritic.py correct --model vnlm/models/vi.pkl.gz --inplace
"""
import os
import io
import re
import gzip
import math
import pickle
import argparse
import tarfile
import unicodedata
from collections import defaultdict

WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)
SYLL_RE = re.compile(r"[a-zà-ỹ]+", re.UNICODE)  # training tokens: letters only


def skeleton(word):
    """Lowercased word with all Vietnamese diacritics stripped (đ -> d)."""
    nfd = unicodedata.normalize("NFD", word.lower())
    base = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return base.replace("đ", "d")


def _match_case(src, repl):
    if src.istitle():
        return repl.title()
    if src.isupper() and len(src) > 1:
        return repl.upper()
    return repl


class DiacriticModel:
    def __init__(self, lam=0.85, keep_boost=2.0, min_count=2):
        self.uni = defaultdict(int)            # syllable -> count
        self.bi = defaultdict(lambda: defaultdict(int))  # prev -> {cur: count}
        self.skel = defaultdict(set)           # skeleton -> {syllable, ...}
        self.N = 0                             # total tokens
        self.lam = lam                         # bigram interpolation weight
        self.keep_boost = keep_boost           # bias toward the OCR reading
        self.min_count = min_count             # ignore rarer candidates

    # ---- training -------------------------------------------------------
    def train_sentence(self, text):
        text = unicodedata.normalize("NFC", text.lower())
        toks = SYLL_RE.findall(text)
        prev = "<s>"
        for w in toks:
            self.uni[w] += 1
            self.bi[prev][w] += 1
            self.skel[skeleton(w)].add(w)
            self.N += 1
            prev = w
        if toks:
            self.bi[prev]["</s>"] += 1

    def train_lines(self, lines, max_sentences=None):
        n = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Leipzig format is "<id>\t<sentence>"; keep only the sentence.
            if "\t" in line:
                line = line.split("\t", 1)[1]
            self.train_sentence(line)
            n += 1
            if max_sentences and n >= max_sentences:
                break
            if n % 500000 == 0:
                print(f"  ...{n:,} sentences")
        return n

    # ---- persistence ----------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "uni": dict(self.uni),
            "bi": {p: dict(c) for p, c in self.bi.items()},
            "skel": {s: sorted(v) for s, v in self.skel.items()},
            "N": self.N,
            "lam": self.lam, "keep_boost": self.keep_boost,
            "min_count": self.min_count,
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(data["lam"], data["keep_boost"], data["min_count"])
        m.uni = defaultdict(int, data["uni"])
        m.bi = defaultdict(lambda: defaultdict(int))
        for p, c in data["bi"].items():
            m.bi[p] = defaultdict(int, c)
        m.skel = defaultdict(set, {s: set(v) for s, v in data["skel"].items()})
        m.N = data["N"]
        return m

    # ---- decoding -------------------------------------------------------
    def _p_uni(self, w):
        return (self.uni.get(w, 0) + 1) / (self.N + len(self.uni) + 1)

    def _logp(self, prev, cur):
        prev_c = self.uni.get(prev, 0)
        p_bi = self.bi.get(prev, {}).get(cur, 0) / prev_c if prev_c else 0.0
        return math.log(self.lam * p_bi + (1 - self.lam) * self._p_uni(cur))

    def _candidates(self, token):
        opts = {c for c in self.skel.get(skeleton(token), ())
                if self.uni.get(c, 0) >= self.min_count}
        opts.add(token)  # always allow keeping the OCR reading
        return list(opts)

    def restore_tokens(self, tokens, frozen=None):
        """Viterbi over same-skeleton candidates for a list of lowercase tokens.

        Indices in `frozen` are never changed (e.g. proper nouns); they still act
        as fixed context for their neighbours.
        """
        if not tokens:
            return []
        frozen = frozen or set()
        bonus = math.log(self.keep_boost)
        cands = [[tokens[i]] if i in frozen else self._candidates(tokens[i])
                 for i in range(len(tokens))]
        scores = {c: self._logp("<s>", c) + (bonus if c == tokens[0] else 0)
                  for c in cands[0]}
        paths = {c: [c] for c in cands[0]}
        for i in range(1, len(tokens)):
            new_scores, new_paths = {}, {}
            for c in cands[i]:
                bp = max(scores, key=lambda p: scores[p] + self._logp(p, c))
                s = scores[bp] + self._logp(bp, c)
                if c == tokens[i]:
                    s += bonus
                new_scores[c], new_paths[c] = s, paths[bp] + [c]
            scores, paths = new_scores, new_paths
        return paths[max(scores, key=scores.get)]

    def restore_line(self, line):
        matches = list(WORD_RE.finditer(line))
        # Only letter-tokens participate in the LM; keep digit tokens verbatim.
        idx = [i for i, m in enumerate(matches) if SYLL_RE.fullmatch(m.group().lower())]
        lowered = [matches[i].group().lower() for i in idx]
        # Freeze proper nouns: capitalized non-initial tokens, and any token
        # glued to a preceding word by a hyphen (e.g. the "chan" in "Ringo-chan").
        frozen = set()
        for k in range(len(idx)):
            cap = k > 0 and matches[idx[k]].group()[:1].isupper()
            start = matches[idx[k]].start()
            hyphen = start > 0 and line[start - 1] == "-"
            if cap or hyphen:
                frozen.add(k)
        restored = self.restore_tokens(lowered, frozen)
        repl = {idx[k]: _match_case(matches[idx[k]].group(), restored[k])
                for k in range(len(idx))}
        out, last = [], 0
        for i, m in enumerate(matches):
            out.append(line[last:m.start()])
            out.append(repl.get(i, m.group()))
            last = m.end()
        out.append(line[last:])
        return "".join(out)

    def restore_text(self, text):
        return "\n".join(self.restore_line(ln) for ln in text.splitlines())


# ---- corpus loading -----------------------------------------------------
# ASS/SSA override blocks {\...} and SRT/HTML tags <i> etc.
SUB_TAG = re.compile(r"\{[^}]*\}|<[^>]*>")


def _strip_sub(text):
    """Remove subtitle markup and split soft line breaks into separate lines."""
    text = SUB_TAG.sub("", text)
    for br in ("\\N", "\\n"):
        text = text.replace(br, "\n")
    text = text.replace("\\h", " ")
    for line in text.split("\n"):
        line = line.strip()
        if line:
            yield line


def _read_text(path):
    """Read a subtitle file, sniffing the BOM for UTF-8/UTF-16 (many fansub
    groups ship UTF-16); fall back to UTF-8 with undecodable bytes ignored."""
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16"
    elif raw[:3] == b"\xef\xbb\xbf":
        enc = "utf-8-sig"
    else:
        enc = "utf-8"
    return raw.decode(enc, errors="ignore")


def iter_srt(lines):
    for line in lines:
        s = line.strip()
        if not s or "-->" in s or s.isdigit():  # blank, timestamp, or index line
            continue
        yield from _strip_sub(line)


def iter_ass(lines):
    for line in lines:
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)  # 9 header fields, then the free-form Text field
        if len(parts) == 10:
            yield from _strip_sub(parts[9])


def iter_corpus_lines(path):
    """Yield clean text lines from a .txt/.gz/.tar.gz, a .srt/.ass/.ssa subtitle
    file, or a directory (recursed; hidden/.git entries skipped). Markup stripped."""
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.startswith("."):  # skip .git and other hidden entries
                continue
            yield from iter_corpus_lines(os.path.join(path, name))
    elif path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and "sentences" in member.name:
                    f = tar.extractfile(member)
                    if f:
                        yield from io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
    elif path.endswith(".gz"):  # plain gzipped text, e.g. OpenSubtitles vi.txt.gz
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            yield from f
    elif path.endswith(".srt"):
        yield from iter_srt(_read_text(path).splitlines())
    elif path.endswith((".ass", ".ssa")):
        yield from iter_ass(_read_text(path).splitlines())
    elif path.endswith(".txt"):
        with io.open(path, encoding="utf-8-sig", errors="ignore") as f:
            yield from f


# ---- CLI ----------------------------------------------------------------
def cmd_train(args):
    if args.grow and os.path.exists(args.model):
        model = DiacriticModel.load(args.model)
        print(f"Loaded existing model ({model.N:,} tokens) to grow.")
    else:
        model = DiacriticModel()
    n = model.train_lines(iter_corpus_lines(args.corpus), args.max_sentences)
    model.save(args.model)
    print(f"Trained on {n:,} sentences -> {model.N:,} tokens, "
          f"{len(model.uni):,} syllable types. Saved {args.model}")


def cmd_correct(args):
    model = DiacriticModel.load(args.model)
    model.keep_boost = args.keep_boost
    src = "./results/pytesseract"
    out_dir = src if args.inplace else "./results/pytesseract_corrected"
    os.makedirs(out_dir, exist_ok=True)
    changed = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith(".txt"):
            continue
        with io.open(os.path.join(src, name), encoding="utf-8") as f:
            text = f.read()
        fixed = model.restore_text(text)
        if fixed != text:
            changed += 1
            if args.verbose:
                for a, b in zip(text.split("\n"), fixed.split("\n")):
                    if a != b:
                        print(f"[{name}]\n  - {a}\n  + {b}")
        with io.open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(fixed)
    print(f"\nCorrected {changed} file(s); output in {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train or grow the model from a corpus")
    t.add_argument("--corpus", required=True, help="file, .tar.gz, or directory")
    t.add_argument("--model", default="vnlm/models/vi.pkl.gz")
    t.add_argument("--grow", action="store_true", help="add to an existing model")
    t.add_argument("--max-sentences", type=int, default=None, dest="max_sentences",
                   help="cap the number of sentences (for a quick/smaller model)")
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("correct", help="restore diacritics on OCR results")
    c.add_argument("--model", default="vnlm/models/vi.pkl.gz")
    c.add_argument("--inplace", action="store_true")
    c.add_argument("--verbose", action="store_true")
    c.add_argument("--keep-boost", type=float, default=14.0, dest="keep_boost",
                   help="bias toward the OCR reading; higher = more conservative")
    c.set_defaults(func=cmd_correct)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
