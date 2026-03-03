# source/build_simp.py

import json
import argparse
from pathlib import Path


# -----------------------
# helpers
# -----------------------
def load_jsonl_dict(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[str(obj["id"])] = obj  # force string id
    return data


# -----------------------
# build
# -----------------------
def build(src_path, llm_path, out_path, num_refs=8, key="simplification"):

    print("[Build] loading:", src_path)
    src_data = load_jsonl_dict(src_path)

    print("[Build] loading:", llm_path)
    llm_data = load_jsonl_dict(llm_path)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as out:

        for sid, src in src_data.items():

            if sid not in llm_data:
                skipped += 1
                continue

            llm = llm_data[sid]

            # ---- read fields (YOUR REAL STRUCTURE) ----
            original = src.get("original", "")
            refs = src.get("references", [])

            base = llm.get("llm_simplified_base", "")
            up = llm.get("llm_simplified_uptodate", "")

            # simple safety check
            if not original or len(refs) != num_refs:
                skipped += 1
                continue

            # ---- build output ----
            items = [{"label": "original", "text": original}]

            for i, r in enumerate(refs):
                items.append({
                    "label": f"ref_{i:02d}",
                    "text": r
                })

            items.append({"label": "llm_base", "text": base})
            items.append({"label": "llm_uptodate", "text": up})

            out.write(json.dumps({
                "id": sid,
                key: items
            }, ensure_ascii=False) + "\n")

            written += 1

    print("\n[Build] DONE")
    print("written :", written)
    print("skipped :", skipped)
    print("output  :", out_path)


# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_jsonl", default="data/tune_sample.jsonl")
    ap.add_argument("--llm_jsonl", default="data/tune_sample_llm.jsonl")
    ap.add_argument("--out", default="data/tune_sample_simp.jsonl")
    ap.add_argument("--num_refs", type=int, default=8)
    ap.add_argument("--key", default="simplification")
    args = ap.parse_args()

    build(
        args.src_jsonl,
        args.llm_jsonl,
        args.out,
        args.num_refs,
        args.key,
    )


if __name__ == "__main__":
    main()