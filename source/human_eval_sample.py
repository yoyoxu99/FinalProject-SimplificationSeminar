# source/human_eval_sample.py

import json
import random
import argparse
from pathlib import Path

import numpy as np
import textstat


# -------------------------
# robust text extraction
# -------------------------
def to_text(x):
    # x may be a string or a dict like {"ref_id": "...", "text": "..."}
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        t = x.get("text")
        if isinstance(t, str):
            return t
        # fallback: first string value
        for v in x.values():
            if isinstance(v, str):
                return v
    return "" if x is None else str(x)


# -------------------------
# FKGL (robust)
# -------------------------
def fkgl(text):
    try:
        v = float(textstat.flesch_kincaid_grade(text))
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


# -------------------------
# quantile bucket (stable 3 bins)
# -------------------------
def quantile_bucket(x):
    # handle constant arrays safely
    q1, q2 = np.nanpercentile(x, [33, 66])
    return np.digitize(x, [q1, q2], right=True)  # -> 0,1,2


# -------------------------
# load + filter refs (refs that equal original are removed)
# also handles refs being dicts like {"ref_id": "...", "text": "..."}
# -------------------------
def load_data(path, key, num_refs):
    originals, base, refs_all, years = [], [], [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            d = json.loads(line)
            items = {x["label"]: x.get("text") for x in d[key]}

            ori = to_text(items.get("original", ""))
            b = to_text(items.get("llm_base", ""))

            if not ori or not b:
                continue

            refs = []
            for i in range(num_refs):
                r = to_text(items.get(f"ref_{i:02d}", ""))
                if r and r != ori:
                    refs.append(r)

            if not refs:
                # if all refs equal original / missing, skip this sample
                continue

            originals.append(ori)
            base.append(b)
            refs_all.append(refs)
            years.append(d.get("year", 0))

    return originals, base, refs_all, np.array(years, dtype=float)


# -------------------------
# compute axes
# -------------------------
def compute_axes(originals, base, years):
    fk_ori = np.array([fkgl(x) for x in originals], dtype=float)
    fk_base = np.array([fkgl(x) for x in base], dtype=float)

    # simplification axis
    delta_fkgl = fk_base - fk_ori

    # modernity axis (centered year)
    delta_year = years - np.nanmean(years)

    return delta_fkgl, delta_year


# -------------------------
# stratified sampling (3×3 grid)
# -------------------------
def stratified_sample(d_fkgl, d_year, n_samples):
    fk_bin = quantile_bucket(d_fkgl)
    yr_bin = quantile_bucket(d_year)

    buckets = [[] for _ in range(9)]
    for i in range(len(d_fkgl)):
        buckets[int(fk_bin[i]) * 3 + int(yr_bin[i])].append(i)

    per_bucket = n_samples // 9

    chosen = []
    remaining = []

    # balanced pass
    for b in buckets:
        random.shuffle(b)
        chosen.extend(b[:per_bucket])
        remaining.extend(b[per_bucket:])

    # fill deficit
    need = n_samples - len(chosen)
    if need > 0 and remaining:
        chosen.extend(random.sample(remaining, min(need, len(remaining))))

    return chosen[:n_samples]


# -------------------------
# build rows for CSV (id, original, cand_a, cand_b)
# cand_a = one reference (random), cand_b = llm_base
# -------------------------
def build_rows(indices, originals, base, refs_all):
    rows = []
    for idx in indices:
        rows.append(
            {
                "id": idx,
                "original": originals[idx],
                "cand_a": random.choice(refs_all[idx]),  # pure text
                "cand_b": base[idx],                     # pure text
            }
        )
    return rows


# -------------------------
# CSV writer (minimal, robust quoting)
# -------------------------
def csv_escape(s: str) -> str:
    if s is None:
        s = ""
    s = str(s)
    # escape double quotes
    s = s.replace('"', '""')
    # wrap in quotes if needed
    if any(c in s for c in [",", '"', "\n", "\r"]):
        return f'"{s}"'
    return s


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("id,original,cand_a,cand_b\n")
        for r in rows:
            line = ",".join(
                [
                    str(r["id"]),
                    csv_escape(r["original"]),
                    csv_escape(r["cand_a"]),
                    csv_escape(r["cand_b"]),
                ]
            )
            f.write(line + "\n")


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--key", default="simplification")
    ap.add_argument("--num_refs", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading data...")
    originals, base, refs_all, years = load_data(args.input, args.key, args.num_refs)
    print("Total usable samples:", len(originals))

    if len(originals) == 0:
        raise SystemExit("No usable samples after filtering refs==original. Check your input format/keys.")

    print("Computing axes...")
    d_fkgl, d_year = compute_axes(originals, base, years)

    print("Stratified sampling...")
    indices = stratified_sample(d_fkgl, d_year, args.n_samples)
    print("Sample size:", len(indices))

    print("Building rows...")
    rows = build_rows(indices, originals, base, refs_all)

    out_path = Path("data/human_eval_samples.csv")
    print("Saving to:", out_path)
    write_csv(out_path, rows)

    print("Done.")


if __name__ == "__main__":
    main()