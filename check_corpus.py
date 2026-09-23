"""Leakage and coverage checks for corpus/ against the fixed 48-case suite."""
import re, json
from collections import Counter
from pathlib import Path
import sys, types
from unittest import mock
sys.modules.setdefault("torch", mock.MagicMock())  # the checker functions never touch torch
from run_evals import load_suite, reject_eval_leakage

def toks(t): return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", t.lower())
suite = load_suite("evals/language_evals.json")
cases = suite["cases"] if isinstance(suite, dict) else suite
for f in sorted(Path("corpus").glob("*.txt")):
    text = f.read_text()
    reject_eval_leakage(text, suite, f.name)          # 1. the course's own gate
    print(f"{f.name}: passed official exact-prompt check")
    words = Counter(toks(text))
    lines = [toks(l) for l in text.splitlines()]
    grams = {tuple(l[i:i+n]) for l in lines for n in (5,6,7,8) for i in range(len(l)-n+1)}
    for c in cases:                                    # 2. stricter: 5+ token runs of prompt+answer
        pa = toks(c["prompt"] + " " + c["answer"])
        hits = [g for n in (5,6,7,8) for i in range(len(pa)-n+1) if (g:=tuple(pa[i:i+n])) in grams]
        if hits: print("  OVERLAP", c["id"], " ".join(max(hits,key=len)))
target = {"grammar","negation"}
allw = Counter(w for f in Path("corpus").glob("*.txt") for w in toks(f.read_text()))
print("\nCoverage (word: count in your files) for target cases:")
for c in cases:
    if c["category"] in target:
        need = set(toks(c["prompt"])) | set(c["choices"])
        low = {w: allw[w] for w in need if allw[w] < 3}
        print(c["id"], c["category"], "OK" if not low else f"LOW/MISSING {low}")
print("\nNew vocabulary types across both files:", len(allw))
