# source/dating.py

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

from prompts import PROMPT_DATING


# ======================
# HELPERS
# ======================

def load_jsonl_as_dict(path: Path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj
    return data


def parse_json_list_or_dict(text: str):
    """
    Minimal robust parse:
    - accept strict JSON list
    - accept strict JSON dict (wrap into list)
    - fallback: extract [...] or {...} via regex
    """
    text = text.strip()

    # strict
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    # regex list
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        obj = json.loads(m.group())
        if isinstance(obj, list):
            return obj

    # regex dict
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        obj = json.loads(m.group())
        if isinstance(obj, dict):
            return [obj]

    raise ValueError(f"JSON parsing failed. Raw output:\n{text}")


def call_dating_batch(client: OpenAI, model_name: str, prompt_template: str, sentences, temperature: float = 0.0):
    joined = "\n".join(f'{item["label"]}: {item["text"]}' for item in sentences)
    prompt = prompt_template.replace("{TEXT}", joined)

    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=temperature,
    )

    return parse_json_list_or_dict(response.output_text)


def main():
    # ======================
    # ARGS (hyperparameters)
    # ======================
    parser = argparse.ArgumentParser()

    parser.add_argument("--orig", type=str, default="tune_sample.jsonl",
                        help="Original file in data/")
    parser.add_argument("--llm", type=str, default="tune_sample_llm.jsonl",
                        help="LLM file in data/")
    parser.add_argument("--output", type=str, default="tune_sample_dated.jsonl",
                        help="Output file in data/")

    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.1)

    # Retry once if output missing labels
    parser.add_argument("--retry", action="store_true",
                        help="Retry once (with minimal repair prompt) if output doesn't contain labels for each sentence")

    args = parser.parse_args()

    # ======================
    # PATHS (match your directory)
    # ======================
    project_root = Path(__file__).resolve().parent.parent
    orig_path = project_root / "data" / args.orig
    llm_path = project_root / "data" / args.llm
    out_path = project_root / "data" / args.output

    print("Orig  :", orig_path)
    print("LLM   :", llm_path)
    print("Output:", out_path)
    print("Model :", args.model, "Temp:", args.temperature, "Sleep:", args.sleep, "Retry:", args.retry)

    client = OpenAI()

    # ======================
    # LOAD
    # ======================
    orig_data = load_jsonl_as_dict(orig_path)
    llm_data = load_jsonl_as_dict(llm_path)

    # ======================
    # MAIN LOOP
    # ======================
    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, _id in enumerate(orig_data, start=1):
            print(f"Processing id={_id} ({idx}/{len(orig_data)})")

            orig = orig_data[_id]
            llm = llm_data.get(_id)

            if llm is None:
                print("Missing LLM output for id=", _id, "-> skip")
                continue

            batch_sentences = []

            # original
            batch_sentences.append({"label": "original", "text": orig.get("original", "")})

            # references
            for ref in orig.get("references", []):
                batch_sentences.append({"label": f'ref_{ref.get("ref_id")}', "text": ref.get("text", "")})

            # llm outputs
            batch_sentences.append({"label": "llm_base", "text": llm.get("llm_simplified_base", "")})
            batch_sentences.append({"label": "llm_uptodate", "text": llm.get("llm_simplified_uptodate", "")})

            expected_labels = [x["label"] for x in batch_sentences]

            # call once
            try:
                dating_result = call_dating_batch(
                    client=client,
                    model_name=args.model,
                    prompt_template=PROMPT_DATING,
                    sentences=batch_sentences,
                    temperature=args.temperature,
                )

                # minimal warning
                got_labels = {x.get("label") for x in dating_result if isinstance(x, dict)}
                if set(expected_labels) != got_labels:
                    missing = set(expected_labels) - got_labels
                    extra = got_labels - set(expected_labels)
                    print(f"WARNING id={_id}: expected {len(expected_labels)} labels, got {len(got_labels)}. "
                          f"Missing={list(missing)[:5]} Extra={list(extra)[:5]}")

                # retry once with a minimal repair prompt (ONLY if --retry)
                if args.retry and not set(expected_labels).issubset(got_labels):
                    time.sleep(args.sleep)

                    joined = "\n".join(f'{item["label"]}: {item["text"]}' for item in batch_sentences)
                    repair_prompt = (
                        PROMPT_DATING.replace("{TEXT}", joined)
                        + "\n\nIMPORTANT: Return EXACTLY one JSON object per label. "
                          "The labels you MUST include are:\n"
                        + "\n".join(expected_labels)
                    )

                    response = client.responses.create(
                        model=args.model,
                        input=repair_prompt,
                        temperature=args.temperature,
                    )
                    dating_result = parse_json_list_or_dict(response.output_text)

                    got_labels = {x.get("label") for x in dating_result if isinstance(x, dict)}
                    if not set(expected_labels).issubset(got_labels):
                        missing = set(expected_labels) - got_labels
                        print(f"WARNING id={_id}: still missing labels after retry: {list(missing)[:10]}")

            except Exception as e:
                print("Error dating id=", _id, "err=", e)
                dating_result = []

            out = {"id": _id, "dating": dating_result}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            time.sleep(args.sleep)

    print("\nFinished!")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()