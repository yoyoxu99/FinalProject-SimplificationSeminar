# source/preprocessing.py

import argparse
import json
import random
from pathlib import Path


def load_lines(input_path):
    """Read non-empty lines."""
    with open(input_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def sample_lines(lines, k, seed):
    """Random sampling."""
    random.seed(seed)

    if k > len(lines):
        raise ValueError(f"Sample size {k} > dataset size {len(lines)}")

    return random.sample(lines, k)


def convert_to_jsonl(lines, output_path):
    """Convert TSV lines → JSONL."""
    with open(output_path, "w", encoding="utf-8") as fout:

        for line in lines:
            cols = line.split("\t")

            if len(cols) < 3:
                continue

            example = {
                "id": int(cols[0]),
                "original": cols[1],
                "references": [
                    {
                        "ref_id": f"{i:02d}",
                        "text": ref
                    }
                    for i, ref in enumerate(cols[2:])
                ]
            }

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")


def main(args):

    project_root = Path(__file__).resolve().parent.parent

    input_path = project_root / "data" / args.input
    output_path = project_root / "data" / args.output

    print("Input:", input_path)
    print("Output:", output_path)

    lines = load_lines(input_path)
    sampled = sample_lines(lines, args.sample_size, args.seed)

    convert_to_jsonl(sampled, output_path)

    print(f"Saved {len(sampled)} samples → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="tune.8turkers.organized.tsv",
        help="Input TSV filename (inside data/)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="tune_sample.jsonl",
        help="Output JSONL filename (inside data/)"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        required=True,
        help="Number of samples"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    main(args)