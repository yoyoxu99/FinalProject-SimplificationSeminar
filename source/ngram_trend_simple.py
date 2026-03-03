# source/ngram_trend_simple.py


import json
import re
import time
from collections import Counter, defaultdict

import requests
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
INPUT = "data/tune_sample_simp.jsonl"
START_YEAR = 2000
END_YEAR = 2019
CORPUS = 26
TOPK = 80
MIN_LEN = 2

MAX_RETRIES = 8
BASE_SLEEP = 1.0
BATCH_SIZE = 8

STOP = {
    "the","a","an","of","to","in","on","at","for","and","or","but","is","are","was","were",
    "be","been","being","it","this","that","as","by","with","from","into","about","than","then","so",
    "can"
}

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")

def tokenize(text: str):
    toks = [t.lower() for t in TOKEN_RE.findall(str(text))]
    toks = [t for t in toks if len(t) >= MIN_LEN and not t.isdigit() and t not in STOP]
    return toks

def diff_ops(orig: str, simp: str):
    o = Counter(tokenize(orig))
    s = Counter(tokenize(simp))
    keep = o & s
    add = s - o
    delete = o - s
    return add, delete, keep

def extract_original_and_base(obj: dict):
    orig, base = None, None
    for item in obj.get("simplification", []):
        if item.get("label") == "original":
            orig = item.get("text")
        elif item.get("label") == "llm_base":
            base = item.get("text")
    return orig, base

def extract_references(obj: dict):
    """
    Returns list[str] of reference texts.
    Handles both:
      {"label":"ref_00", "text":"..."}
    and
      {"label":"ref_00", "text":{"ref_id":"00","text":"..."}}
    """
    refs = []
    for item in obj.get("simplification", []):
        label = item.get("label", "")
        if not label.startswith("ref_"):
            continue
        t = item.get("text")
        if isinstance(t, dict):
            t = t.get("text")
        if isinstance(t, str) and t.strip():
            refs.append(t)
    return refs

def update_vocab_from_pair(orig, cand, add_vocab, del_vocab, keep_vocab):
    add, delete, keep = diff_ops(orig, cand)
    add_vocab.update(add)
    del_vocab.update(delete)
    keep_vocab.update(keep)

# =====================
# Robust request with retries
# =====================
def request_with_backoff(session: requests.Session, url: str, params: dict):
    sleep = BASE_SLEEP
    for _ in range(MAX_RETRIES):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            time.sleep(sleep)
            sleep *= 2
            continue
        r.raise_for_status()
    raise RuntimeError("Too many 429 responses; try smaller TOPK/BATCH_SIZE or rerun later.")

def get_ngram_batch(session: requests.Session, words):
    url = "https://books.google.com/ngrams/json"
    params = {
        "content": ",".join(words),
        "year_start": START_YEAR,
        "year_end": END_YEAR,
        "corpus": CORPUS,
        "smoothing": 0,
    }
    r = request_with_backoff(session, url, params)
    data = r.json()

    out = {}
    for item in data:
        w = item.get("ngram")
        series = item.get("timeseries", [])
        if not w or not series:
            continue
        years = list(range(START_YEAR, START_YEAR + len(series)))
        out[w] = dict(zip(years, series))
    return out

def collect_curve(words):
    session = requests.Session()
    yearly_vals = defaultdict(list)

    for i in range(0, len(words), BATCH_SIZE):
        batch = words[i:i + BATCH_SIZE]
        batch_ts = get_ngram_batch(session, batch)

        for w in batch:
            ts = batch_ts.get(w)
            if not ts:
                continue
            for y, v in ts.items():
                yearly_vals[y].append(v)

        time.sleep(0.5)

    curve = {y: (sum(vs) / len(vs)) for y, vs in yearly_vals.items() if vs}
    return curve

# =====================
# MAIN: build vocabs for BASE and REFS
# =====================
base_add = Counter(); base_del = Counter(); base_keep = Counter()
ref_add  = Counter(); ref_del  = Counter(); ref_keep  = Counter()

used_base = 0
used_refs = 0

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        orig, base = extract_original_and_base(obj)
        if orig is None:
            continue

        # ---- BASE ----
        if base is not None:
            update_vocab_from_pair(orig, base, base_add, base_del, base_keep)
            used_base += 1

        # ---- REFERENCES ----
        refs = extract_references(obj)
        for rtext in refs:
            update_vocab_from_pair(orig, rtext, ref_add, ref_del, ref_keep)
            used_refs += 1

print(f"BASE pairs used: {used_base}")
print(f"REF pairs used (orig x refs): {used_refs}")

# pick top words
base_add_words = [w for w, _ in base_add.most_common(TOPK)]
base_del_words = [w for w, _ in base_del.most_common(TOPK)]
base_keep_words = [w for w, _ in base_keep.most_common(TOPK)]

ref_add_words = [w for w, _ in ref_add.most_common(TOPK)]
ref_del_words = [w for w, _ in ref_del.most_common(TOPK)]
ref_keep_words = [w for w, _ in ref_keep.most_common(TOPK)]

print(f"Using TOPK={TOPK}, BATCH_SIZE={BATCH_SIZE}")
print("Fetching BASE curves...")
base_add_curve = collect_curve(base_add_words)
base_del_curve = collect_curve(base_del_words)
base_keep_curve = collect_curve(base_keep_words)

print("Fetching REF curves...")
ref_add_curve = collect_curve(ref_add_words)
ref_del_curve = collect_curve(ref_del_words)
ref_keep_curve = collect_curve(ref_keep_words)

years = list(range(START_YEAR, END_YEAR + 1))

# =====================
# PLOT (6 curves)
# =====================
plt.figure(figsize=(10, 5))

plt.plot(years, [base_add_curve.get(y, 0.0) for y in years], label="BASE-ADD")
plt.plot(years, [base_del_curve.get(y, 0.0) for y in years], label="BASE-DELETE")
plt.plot(years, [base_keep_curve.get(y, 0.0) for y in years], label="BASE-KEEP")

plt.plot(years, [ref_add_curve.get(y, 0.0) for y in years], linestyle="--", label="REF-ADD")
plt.plot(years, [ref_del_curve.get(y, 0.0) for y in years], linestyle="--", label="REF-DELETE")
plt.plot(years, [ref_keep_curve.get(y, 0.0) for y in years], linestyle="--", label="REF-KEEP")

plt.xlabel("Year")
plt.ylabel("Mean frequency (Google Books Ngrams)")
plt.title("ADD/DELETE/KEEP frequency trends: LLM Base vs References")
plt.legend()
plt.tight_layout()
plt.show()
