# source/prompts.py

PROMPT_DATING = """
You are a historical linguistics expert specializing in diachronic language change and the current year is 2025.

Multiple sentences are given below. Each line is formatted as:
label: sentence

Your task is to estimate the MOST LIKELY TIME SPAN for EACH sentence.

Requirements:
- Base your decision on linguistic and stylistic evidence.
- Do NOT explain your reasoning.
- Each prediction MUST be between 2000 and 2025.
- Each span should normally cover at most 20 years.
- Output STRICT JSON only (no extra text).
- Return EXACTLY ONE item per label.

Format:
[
  {"label": "...", "start_year": YYYY, "end_year": YYYY}
]

Text:
{TEXT}
"""


PROMPT_SIMP_BASE = """
You are required to simplify the original sentence.

Requirements:
- Use simpler concepts, words, or phrases.
- Keep the meaning the same.
- Output ONLY the simplified sentence.

Original sentence:
{TEXT}
"""

PROMPT_SIMP_UPTODATE = """
You are required to simplify the original sentence.

Requirements:
- The current year is 2025, use simpler and up-to-date contemporary concepts, words, or phrases.
- Keep the meaning the same.
- Output ONLY the simplified sentence.

Original sentence:
{TEXT}
"""