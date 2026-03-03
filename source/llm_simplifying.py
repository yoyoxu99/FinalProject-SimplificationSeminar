# source/llm_simplifying.py

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

from prompts import PROMPT_SIMP_BASE, PROMPT_SIMP_UPTODATE


def simplify_with_prompt(client: OpenAI, model: str, temperature: float, text: str, prompt_template: str) -> str:
    prompt = prompt_template.replace("{TEXT}", text)

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )

    return response.output_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="tune_sample.jsonl",
                        help="Input JSONL filename (inside data/)")
    parser.add_argument("--output", type=str, default="tune_sample_llm.jsonl",
                        help="Output JSONL filename (inside data/)")
    parser.add_argument("--model", type=str, default="gpt-5.2",
                        help="Model name")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature")
    parser.add_argument("--sleep", type=float, default=0.1,
                        help="Sleep time between requests (seconds)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / args.input
    output_path = project_root / "data" / args.output

    print("Input:", input_path)
    print("Output:", output_path)
    print("Model:", args.model, "Temp:", args.temperature, "Sleep:", args.sleep)

    client = OpenAI()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin, start=1):
            data = json.loads(line)

            _id = data["id"]
            original_text = data["original"]

            print(f"Processing id={_id} ({i})")

            # Base simplification
            try:
                simp_base = simplify_with_prompt(
                    client, args.model, args.temperature, original_text, PROMPT_SIMP_BASE
                )
            except Exception as e:
                print("Error (base):", e)
                simp_base = ""

            time.sleep(args.sleep)

            # Up-to-date simplification
            try:
                simp_uptodate = simplify_with_prompt(
                    client, args.model, args.temperature, original_text, PROMPT_SIMP_UPTODATE
                )
            except Exception as e:
                print("Error (up-to-date):", e)
                simp_uptodate = ""

            # Generation-only output
            out = {
                "id": _id,
                "llm_simplified_base": simp_base,
                "llm_simplified_uptodate": simp_uptodate,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)

    print("\nFinished!")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()