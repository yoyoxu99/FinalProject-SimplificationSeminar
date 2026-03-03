# source/date_eval.py


import json
import argparse
import numpy as np
from scipy.stats import ttest_rel

# -----------------------
# helper
# -----------------------
def midpoint(item):
    return (item["start_year"] + item["end_year"]) / 2

def closer_to_target(target_year, a, b):
    return abs(target_year - a) < abs(target_year - b)

def ratio(target_year, a, b):
    return np.mean([closer_to_target(target_year, x, y) for x, y in zip(a, b)])

def mean_pm_std(x):
    # sample std (ddof=1) is standard for reporting
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"

def fmt_years(x):
    # +x.xx years
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f} years"

def fmt_pct(x):
    # 0.9933 -> 99.3%
    return f"{x * 100:.1f}%"

# -----------------------
# main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/tune_sample_dated.jsonl")
    parser.add_argument("--target_year", type=float, default=2025.0)
    parser.add_argument("--num_refs", type=int, default=8)
    parser.add_argument("--sanity_n", type=int, default=0)
    parser.add_argument("--p_threshold", type=float, default=0.001)
    args = parser.parse_args()

    INPUT_FILE = args.input

    # -----------------------
    # containers
    # -----------------------
    original_years = []
    reference_years = []
    llm_base_years = []
    llm_up_years = []
    kept_ids = []

    # -----------------------
    # load data
    # -----------------------
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            items = data["dating"]

            Yo = None
            Yb = None
            Yu = None
            refs_by_label = {}

            for item in items:
                year = midpoint(item)
                label = item["label"]

                if label == "original":
                    Yo = year
                elif label.startswith("ref_"):
                    refs_by_label[label] = year
                elif label == "llm_base":
                    Yb = year
                elif label == "llm_uptodate":
                    Yu = year

            # enforce: must have all refs ref_00..ref_{num_refs-1}
            ref_labels = [f"ref_{i:02d}" for i in range(args.num_refs)]
            if Yo is None or Yb is None or Yu is None:
                continue
            if not all(lbl in refs_by_label for lbl in ref_labels):
                continue

            refs = [refs_by_label[lbl] for lbl in ref_labels]
            Yr = np.mean(refs)

            original_years.append(Yo)
            reference_years.append(Yr)
            llm_base_years.append(Yb)
            llm_up_years.append(Yu)
            kept_ids.append(data.get("id"))

    # convert to numpy
    original_years = np.array(original_years, dtype=float)
    reference_years = np.array(reference_years, dtype=float)
    llm_base_years = np.array(llm_base_years, dtype=float)
    llm_up_years = np.array(llm_up_years, dtype=float)

    print(f"\nValid N = {len(original_years)}")

    # -----------------------
    # MEAN YEARS (mean ± std)
    # -----------------------
    print("\n===== MEAN YEARS =====")
    print("Original :", mean_pm_std(original_years))
    print("Reference:", mean_pm_std(reference_years))
    print("LLM Base :", mean_pm_std(llm_base_years))
    print("LLM UpToDate :", mean_pm_std(llm_up_years))

    # -----------------------
    # MODERNITY GAIN
    # -----------------------
    base_minus_orig = (llm_base_years - original_years).mean()
    up_minus_orig = (llm_up_years - original_years).mean()
    up_minus_base = (llm_up_years - llm_base_years).mean()

    print("\n===== MODERNITY GAIN =====")
    print("Base \u2212 Original :", fmt_years(base_minus_orig))
    print("UpToDate \u2212 Original :", fmt_years(up_minus_orig))
    print("UpToDate \u2212 Base :", fmt_years(up_minus_base))

    # -----------------------
    # SIGNIFICANCE TEST
    # -----------------------
    p1 = ttest_rel(llm_base_years, original_years).pvalue
    p2 = ttest_rel(llm_up_years, original_years).pvalue
    p3 = ttest_rel(llm_up_years, llm_base_years).pvalue
    all_sig = (p1 < args.p_threshold) and (p2 < args.p_threshold) and (p3 < args.p_threshold)

    print("\n===== SIGNIFICANCE TEST =====")
    if all_sig:
        print(f"(all p < {args.p_threshold})")
    else:
        print(f"Base vs Original: p={p1:.3g}")
        print(f"UpToDate vs Original: p={p2:.3g}")
        print(f"UpToDate vs Base: p={p3:.3g}")

    # -----------------------
    # MORE MODERN RATIO
    # -----------------------
    r_bo = ratio(args.target_year, llm_base_years, original_years)
    r_uo = ratio(args.target_year, llm_up_years, original_years)
    r_ub = ratio(args.target_year, llm_up_years, llm_base_years)

    print("\n===== MORE MODERN RATIO =====")
    print(f"Target year: {args.target_year:.1f}")
    print("Base > Original :", fmt_pct(r_bo))
    print("UpToDate > Original :", fmt_pct(r_uo))
    print("UpToDate > Base :", fmt_pct(r_ub))

    # -----------------------
    # sanity check (optional)
    # -----------------------
    if args.sanity_n > 0:
        print(f"\n===== FIRST {min(args.sanity_n, len(original_years))} ITEMS (SANITY CHECK) =====")
        for i in range(min(args.sanity_n, len(original_years))):
            print(
                f"id={kept_ids[i]} | "
                f"Yo={original_years[i]:.1f} Yr={reference_years[i]:.1f} "
                f"Yb={llm_base_years[i]:.1f} Yu={llm_up_years[i]:.1f} | "
                f"Yu-Yb={llm_up_years[i]-llm_base_years[i]:.1f}"
            )

if __name__ == "__main__":
    main()