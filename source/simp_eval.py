# simp_eval.py

import json
import argparse
import numpy as np
import re
import unicodedata
from sacremoses import MosesTokenizer, MosesDetokenizer

MT = MosesTokenizer(lang="en")
MD = MosesDetokenizer(lang="en")


# ------------------ utils ------------------
def to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "output", "prediction", "pred"):
            v = x.get(k)
            if isinstance(v, str):
                return v
        for v in x.values():
            if isinstance(v, str):
                return v
    return str(x)


def norm(s):
    s = unicodedata.normalize("NFKC", to_text(s))
    return re.sub(r"\s+", " ", s).strip()


def detok_if_needed(s):
    s = norm(s)
    # minimal heuristic: common tokenization artifacts
    if (" ." in s) or (" ," in s) or (" n't" in s) or (" )" in s) or ("( " in s):
        return MD.detokenize(s.split())
    return s


def tok_len(s):
    return len(MT.tokenize(norm(s), return_str=False))


def ttr(s):
    toks = MT.tokenize(norm(s), return_str=False)
    return (len(set(toks)) / len(toks)) if toks else 0.0


def prep_texts(xs):
    return [detok_if_needed(x) for x in xs]


def prep_refs(refs_all):
    return [[detok_if_needed(r) for r in refs] for refs in refs_all]


# ------------------ load ------------------
def load_eval(path, key, num_refs):
    originals, base, refs_all = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items = data[key]
            by_label = {x["label"]: x.get("text") for x in items}

            originals.append(norm(by_label["original"]))
            base.append(norm(by_label["llm_base"]))
            refs_all.append([norm(by_label[f"ref_{i:02d}"]) for i in range(num_refs)])

    return originals, base, refs_all


def refs_from_originals(originals):
    return [[o] for o in originals]  # [N] -> [N][1]


# ------------------ FKGL ------------------
def avg_fkgl(texts):
    import textstat
    texts = prep_texts(texts)
    vals = []
    for t in texts:
        try:
            vals.append(float(textstat.flesch_kincaid_grade(t)))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else 0.0


def avg_fkgl_refs(refs_all):
    return avg_fkgl([r for refs in refs_all for r in refs])


# ------------------ SARI (EASSE) ------------------
def sari_easse_multi(originals, system, refs_all, tokenizer="13a", lowercase=True):
    from easse.sari import get_corpus_sari_operation_scores
    refs_sents = list(map(list, zip(*refs_all)))
    add, keep, delete = get_corpus_sari_operation_scores(
        orig_sents=originals,
        sys_sents=system,
        refs_sents=refs_sents,
        lowercase=lowercase,
        tokenizer=tokenizer,
        legacy=False,
        use_f1_for_deletion=True,
        use_paper_version=False,
    )
    sari = (add + keep + delete) / 3.0
    return float(sari), float(add), float(keep), float(delete)


# ------------------ BLEU ------------------
def bleu(cands, refs_all, force=False):
    import sacrebleu
    cands = prep_texts(cands)
    refs_all = prep_refs(refs_all)
    refs_T = list(map(list, zip(*refs_all)))
    return float(sacrebleu.corpus_bleu(cands, refs_T, force=force).score)


# ------------------ BERTScore (multi-ref mean) ------------------
def make_bertscorer(model_type="roberta-large", batch_size=32, device=None):
    from bert_score import BERTScorer
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return BERTScorer(
        model_type=model_type,
        lang="en",
        rescale_with_baseline=True,
        device=device,
        batch_size=batch_size,
    )


def bertscore_multimean(scorer, cands, refs_all):
    cands = prep_texts(cands)
    refs_all = prep_refs(refs_all)
    per_sample = []
    for cand, refs in zip(cands, refs_all):
        if not refs:
            per_sample.append(0.0)
            continue
        scores = []
        for ref in refs:
            _, _, f1 = scorer.score([cand], [ref])
            scores.append(float(f1.detach().cpu().item()))
        per_sample.append(float(np.mean(scores)))
    return float(np.mean(per_sample)) if per_sample else 0.0


# ------------------ coverage + filtered refs ------------------
def coverage_equal_original(originals, base, refs_all):
    N = len(originals)
    R = len(refs_all[0]) if N else 0
    any_ref = 0
    ref_counts = [0] * R

    for o, refs in zip(originals, refs_all):
        flags = [norm(r) == norm(o) for r in refs]
        any_ref += int(any(flags))
        for i, f in enumerate(flags):
            ref_counts[i] += int(f)

    base_eq = sum(1 for o, b in zip(originals, base) if norm(o) == norm(b))

    print("\n===== COVERAGE (text == original; after normalize) =====")
    print(f"Any ref == original      : {any_ref}/{N} = {any_ref/N:.3f}")
    for i, c in enumerate(ref_counts):
        print(f"ref_{i:02d} == original     : {c}/{N} = {c/N:.3f}")
    print(f"LLM Base == original     : {base_eq}/{N} = {base_eq/N:.3f}")


def filter_refs_equal_original(originals, refs_all):
    refs_filt = []
    total_dropped = 0
    dropped_all = 0

    for o, refs in zip(originals, refs_all):
        keep = [r for r in refs if norm(r) != norm(o)]
        total_dropped += (len(refs) - len(keep))
        if not keep:
            dropped_all += 1
            keep = [o]  # fallback keep shape valid
        refs_filt.append(keep)

    return refs_filt, total_dropped, dropped_all


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/tune_sample_simp.jsonl")
    ap.add_argument("--key", default="simplification")
    ap.add_argument("--num_refs", type=int, default=8)

    ap.add_argument("--sari_tokenizer", default="13a", choices=["13a", "intl", "moses", "plain"])
    ap.add_argument("--sari_no_lowercase", action="store_true")

    ap.add_argument("--bleu_force", action="store_true")

    ap.add_argument("--bertscore_model", default="roberta-large")
    ap.add_argument("--bertscore_batch_size", type=int, default=32)
    ap.add_argument("--bertscore_device", default=None)
    args = ap.parse_args()

    originals, base, refs_all = load_eval(args.input, args.key, args.num_refs)
    flat_refs = [r for refs in refs_all for r in refs]
    refs_ori = refs_from_originals(originals)

    refs_filt, dropped_refs, dropped_all = filter_refs_equal_original(originals, refs_all)

    # LENGTH/TTR
    print("\n===== LENGTH (avg tokens; pooled refs) =====")
    print("Original:", round(np.mean([tok_len(x) for x in originals]), 1))
    print("Reference:", round(np.mean([tok_len(x) for x in flat_refs]), 1))
    print("LLM Base:", round(np.mean([tok_len(x) for x in base]), 1))

    print("\n===== TTR =====")
    print("Original:", round(np.mean([ttr(x) for x in originals]), 3))
    print("Reference:", round(np.mean([ttr(x) for x in flat_refs]), 3))
    print("LLM Base:", round(np.mean([ttr(x) for x in base]), 3))

    # SARI
    print("\n===== SARI (EASSE; multi-ref; 0-100) =====")
    lc = not args.sari_no_lowercase
    sari, add, keep, delete = sari_easse_multi(
        originals, base, refs_all, tokenizer=args.sari_tokenizer, lowercase=lc
    )
    print(f"LLM Base: SARI={sari:.3f} | add={add:.3f} keep={keep:.3f} delete={delete:.3f}")

    # FKGL
    print("\n===== FKGL (Flesch-Kincaid Grade Level; lower=easier) =====")
    fk_ori = avg_fkgl(originals)
    fk_ref = avg_fkgl_refs(refs_all)
    fk_base = avg_fkgl(base)
    print("FKGL (Original):", round(fk_ori, 2))
    print("FKGL (References, pooled):", round(fk_ref, 2))
    print("FKGL (LLM Base):", round(fk_base, 2))

    print("\n===== FKGL deltas vs Original =====")
    print("ΔFKGL (References - Original):", round(fk_ref - fk_ori, 2))
    print("ΔFKGL (LLM Base - Original):", round(fk_base - fk_ori, 2))

    # BLEU
    print("\n===== BLEU (sacreBLEU) =====")
    print("BLEU (Base vs Refs):", round(bleu(base, refs_all, force=args.bleu_force), 2))
    print("BLEU (Original vs Refs):", round(bleu(originals, refs_all, force=args.bleu_force), 2))
    print("BLEU (Original vs Refs (filtered)):", round(bleu(originals, refs_filt, force=args.bleu_force), 2))
    print("BLEU (Base vs Original):", round(bleu(base, refs_ori, force=args.bleu_force), 2))

    # coverage + filtered stats
    coverage_equal_original(originals, base, refs_all)
    print(f"[Filtered refs] total dropped refs = {dropped_refs}; "
          f"samples with all refs dropped (fallback to original) = {dropped_all}/{len(originals)}")

    # BERTScore
    print("\n===== BERTScore (multi-ref MEAN; F1) =====")
    scorer = make_bertscorer(args.bertscore_model, args.bertscore_batch_size, args.bertscore_device)
    print("BERTScore (Base vs Refs):", round(bertscore_multimean(scorer, base, refs_all), 3))
    print("BERTScore (Original vs Refs):", round(bertscore_multimean(scorer, originals, refs_all), 3))
    print("BERTScore (Original vs Refs (filtered)):", round(bertscore_multimean(scorer, originals, refs_filt), 3))
    print("BERTScore (Base vs Original):", round(bertscore_multimean(scorer, base, refs_ori), 3))


if __name__ == "__main__":
    main()