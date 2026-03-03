# source/human_eval.py

import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd


NEED_COLS_MIN = ["id", "type", "simplicity", "uptodateness"]


# -----------------------
# Helpers: loading/cleaning
# -----------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = clean_columns(df)

    missing = [c for c in NEED_COLS_MIN if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}. Detected: {df.columns.tolist()}")

    df["id"] = df["id"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    # numeric columns
    df["simplicity"] = pd.to_numeric(df["simplicity"], errors="coerce")
    df["uptodateness"] = pd.to_numeric(df["uptodateness"], errors="coerce")

    return df


def candidates_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["type"] != "original"].copy()


# -----------------------
# Significance test (paired t-test)
# -----------------------
def paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Paired t-test without scipy (uses normal approximation for p-value if df large).
    For small n, this is still OK as a quick report; if you want exact t CDF, use scipy.
    Returns: (t_stat, p_value_two_sided)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        return float("nan"), float("nan")

    d = a - b
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    if sd_d == 0:
        if mean_d == 0:
            return 0.0, 1.0
        return float("inf"), 0.0

    t = mean_d / (sd_d / np.sqrt(n))

    # p-value: use survival function approximation via normal if n>=30
    if n >= 30:
        # normal approx
        from math import erf, sqrt

        z = abs(t)
        # two-sided p for normal:
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    else:
        # fallback: conservative normal approx for small n
        from math import erf, sqrt

        z = abs(t)
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))

    return float(t), float(p)


def system_paired_tests(df: pd.DataFrame, sys_a="cand_a", sys_b="cand_b") -> dict:
    """
    Align by id between cand_a and cand_b within a single annotator dataframe.
    """
    sub = df[df["type"].isin([sys_a, sys_b])].copy()
    # pivot to have columns per system
    piv_s = sub.pivot_table(index="id", columns="type", values="simplicity", aggfunc="mean")
    piv_u = sub.pivot_table(index="id", columns="type", values="uptodateness", aggfunc="mean")

    if sys_a not in piv_s.columns or sys_b not in piv_s.columns:
        return {"n": 0, "simplicity": (np.nan, np.nan), "uptodateness": (np.nan, np.nan)}

    a_s, b_s = piv_s[sys_a].to_numpy(), piv_s[sys_b].to_numpy()
    a_u, b_u = piv_u[sys_a].to_numpy(), piv_u[sys_b].to_numpy()

    # paired tests (a - b)
    t_s, p_s = paired_ttest(a_s, b_s)
    t_u, p_u = paired_ttest(a_u, b_u)

    n = int(np.isfinite(a_s).sum() and np.isfinite(b_s).sum())
    # better n: number of paired finite items
    n = int(np.sum(np.isfinite(a_s) & np.isfinite(b_s)))

    return {"n": n, "simplicity": (t_s, p_s), "uptodateness": (t_u, p_u)}


# -----------------------
# IAA metrics
# -----------------------
def quadratic_weighted_kappa(r1: np.ndarray, r2: np.ndarray, min_rating=1, max_rating=5) -> float:
    """
    Quadratic Weighted Kappa (Cohen), from scratch.
    Assumes integer ordinal ratings in [min_rating, max_rating].
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    mask = np.isfinite(r1) & np.isfinite(r2)
    r1, r2 = r1[mask].astype(int), r2[mask].astype(int)

    if len(r1) == 0:
        return float("nan")

    k = max_rating - min_rating + 1
    # confusion matrix O
    O = np.zeros((k, k), dtype=float)
    for a, b in zip(r1, r2):
        if a < min_rating or a > max_rating or b < min_rating or b > max_rating:
            continue
        O[a - min_rating, b - min_rating] += 1.0

    # expected matrix E
    hist1 = O.sum(axis=1)
    hist2 = O.sum(axis=0)
    N = O.sum()
    if N == 0:
        return float("nan")
    E = np.outer(hist1, hist2) / N

    # weight matrix W (quadratic)
    W = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            W[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return 1.0 - num / den


def krippendorff_alpha_ordinal(r1: np.ndarray, r2: np.ndarray, min_rating=1, max_rating=5) -> float:
    """
    Krippendorff's alpha for ordinal data, 2 annotators only, from scratch.
    Uses ordinal distance: delta(a,b) = (a-b)^2 / (k-1)^2  (common choice for ordinal alpha)
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    mask = np.isfinite(r1) & np.isfinite(r2)
    r1, r2 = r1[mask], r2[mask]
    if len(r1) == 0:
        return float("nan")

    k = max_rating - min_rating + 1

    # Observed disagreement Do: mean distance over all paired items
    Do = np.mean(((r1 - r2) ** 2) / ((k - 1) ** 2))

    # Expected disagreement De: computed from pooled distribution
    pooled = np.concatenate([r1, r2])
    pooled = pooled[(pooled >= min_rating) & (pooled <= max_rating)]
    if len(pooled) == 0:
        return float("nan")

    # frequency of each rating
    vals, counts = np.unique(pooled.astype(int), return_counts=True)
    freq = dict(zip(vals.tolist(), counts.tolist()))
    N = len(pooled)

    # De = sum_{c,c'} p(c)p(c') * delta(c,c')
    # where p(c)=freq(c)/N
    De = 0.0
    for c in range(min_rating, max_rating + 1):
        pc = freq.get(c, 0) / N
        for c2 in range(min_rating, max_rating + 1):
            pc2 = freq.get(c2, 0) / N
            De += pc * pc2 * (((c - c2) ** 2) / ((k - 1) ** 2))

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def align_for_iaa(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Align two annotator dfs by (id,type), candidates-only.
    """
    c1 = candidates_only(df1)[["id", "type", "simplicity", "uptodateness"]].copy()
    c2 = candidates_only(df2)[["id", "type", "simplicity", "uptodateness"]].copy()

    merged = c1.merge(
        c2,
        on=["id", "type"],
        how="inner",
        suffixes=("_a1", "_a2"),
    )
    return merged


# -----------------------
# Reporting
# -----------------------
def report_basic(df: pd.DataFrame, name: str):
    cand = candidates_only(df)
    print(f"\n===== {name}: OVERALL (candidates only) =====")
    print("N rows:", len(cand))
    print("simplicity mean:", round(cand["simplicity"].mean(), 3))
    print("uptodateness mean:", round(cand["uptodateness"].mean(), 3))

    print(f"\n===== {name}: BY SYSTEM =====")
    system_means = cand.groupby("type")[["simplicity", "uptodateness"]].mean().round(3)
    print(system_means)

    corr = cand["simplicity"].corr(cand["uptodateness"])
    print("\nCorrelation (simplicity vs uptodateness):", None if pd.isna(corr) else round(corr, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", required=True, help="Annotator 1 CSV (must contain id,type,simplicity,uptodateness)")
    ap.add_argument("--csv2", default=None, help="Annotator 2 CSV (optional, for IAA + aggregated reporting)")
    ap.add_argument("--sys_a", default="cand_a", help="System name A in column 'type' (default: cand_a)")
    ap.add_argument("--sys_b", default="cand_b", help="System name B in column 'type' (default: cand_b)")
    ap.add_argument("--out_merged", default=None, help="Optional: save aligned merged CSV for IAA/debugging")
    args = ap.parse_args()

    df1 = load_csv(args.csv1)
    report_basic(df1, "Annotator1")

    # significance within annotator1
    tests1 = system_paired_tests(df1, sys_a=args.sys_a, sys_b=args.sys_b)
    print(f"\n===== Annotator1: PAIRED t-test ({args.sys_a} vs {args.sys_b}) =====")
    print("paired N (by id):", tests1["n"])
    t_s, p_s = tests1["simplicity"]
    t_u, p_u = tests1["uptodateness"]
    print(f"simplicity:   t={t_s:.3f}, p≈{p_s:.4f}")
    print(f"uptodateness: t={t_u:.3f}, p≈{p_u:.4f}")

    if args.csv2:
        df2 = load_csv(args.csv2)
        report_basic(df2, "Annotator2")

        # significance within annotator2
        tests2 = system_paired_tests(df2, sys_a=args.sys_a, sys_b=args.sys_b)
        print(f"\n===== Annotator2: PAIRED t-test ({args.sys_a} vs {args.sys_b}) =====")
        print("paired N (by id):", tests2["n"])
        t_s2, p_s2 = tests2["simplicity"]
        t_u2, p_u2 = tests2["uptodateness"]
        print(f"simplicity:   t={t_s2:.3f}, p≈{p_s2:.4f}")
        print(f"uptodateness: t={t_u2:.3f}, p≈{p_u2:.4f}")

        # IAA
        merged = align_for_iaa(df1, df2)
        if args.out_merged:
            merged.to_csv(args.out_merged, index=False)
            print("\nSaved merged alignment for IAA to:", args.out_merged)

        print("\n===== IAA (Annotator1 vs Annotator2; candidates; matched by id+type) =====")
        print("Matched items:", len(merged))

        # QWK
        qwk_s = quadratic_weighted_kappa(merged["simplicity_a1"].values, merged["simplicity_a2"].values)
        qwk_u = quadratic_weighted_kappa(merged["uptodateness_a1"].values, merged["uptodateness_a2"].values)
        print(f"QWK simplicity:   {qwk_s:.3f}")
        print(f"QWK uptodateness: {qwk_u:.3f}")

        # Krippendorff's alpha (ordinal)
        a_s = krippendorff_alpha_ordinal(merged["simplicity_a1"].values, merged["simplicity_a2"].values)
        a_u = krippendorff_alpha_ordinal(merged["uptodateness_a1"].values, merged["uptodateness_a2"].values)
        print(f"Alpha (ordinal) simplicity:   {a_s:.3f}")
        print(f"Alpha (ordinal) uptodateness: {a_u:.3f}")

        # Optional: aggregate (mean of annotators) and rerun system comparison
        agg = merged.copy()
        agg["simplicity"] = agg[["simplicity_a1", "simplicity_a2"]].mean(axis=1)
        agg["uptodateness"] = agg[["uptodateness_a1", "uptodateness_a2"]].mean(axis=1)

        print("\n===== AGGREGATED (mean of 2 annotators; matched items only) =====")
        overall_s = agg["simplicity"].mean()
        overall_u = agg["uptodateness"].mean()
        print("simplicity mean:", round(overall_s, 3))
        print("uptodateness mean:", round(overall_u, 3))

        by_sys = agg.groupby("type")[["simplicity", "uptodateness"]].mean().round(3)
        print("\nBY SYSTEM (aggregated):")
        print(by_sys)

        # aggregated paired test
        tests_agg = system_paired_tests(
            agg.rename(columns={"simplicity": "simplicity", "uptodateness": "uptodateness"})[["id", "type", "simplicity", "uptodateness"]],
            sys_a=args.sys_a,
            sys_b=args.sys_b,
        )
        print(f"\n===== AGGREGATED: PAIRED t-test ({args.sys_a} vs {args.sys_b}) =====")
        print("paired N (by id):", tests_agg["n"])
        t_sa, p_sa = tests_agg["simplicity"]
        t_ua, p_ua = tests_agg["uptodateness"]
        print(f"simplicity:   t={t_sa:.3f}, p≈{p_sa:.4f}")
        print(f"uptodateness: t={t_ua:.3f}, p≈{p_ua:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
