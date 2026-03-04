# Diachronic Analysis of LLM-based Text Simplification

Author: You Xu  
Course: Simplification Seminar
LMU Munich  

---

## 1. Project Overview

This project investigates whether LLMs produce text simplifications that are not only simpler but also more up-to-date in language usage.

We propose a diachronic evaluation framework that combines:

- Automatic simplification metrics (SARI, BLEU, BERTScore)
- Human evaluation (simplicity & up-to-dateness)
- Automatic document dating (LLM-based)
- N-gram frequency evolution analysis (2000–2019)


---


## 2. Repository Structure

```
.
├── README.md
├── data/
│   ├──tune.8turkers.organized.tsv # raw data from TurkCorpus (Xu et al., 2016)
│   ├── tune_sample.jsonl          # sampled data = 600 original sentences + 600*8 human references
│   ├── tune_sample_llm.jsonl      # LLM simplified sentences = 600 LLM BASE + 600 LLM UPTODATE
│   ├── tune_sample_simp.jsonl     # merge all sentences = original + references + LLM
│   ├── tune_sample_dated.jsonl    # sentences with date labels
│   ├── human_eval_sample.jsonl    # human annotation samples = 60 original + 60 references + 60 LLM BASE
│   ├── annotation_sheet_01.csv    # annotations from annotator 1
│   └── annotation_sheet_02.csv    # annotations from annotator 2
│
├── human_annotations/ # detailed annotation guidelines + annotation template + raw annotations 
│   ├── annotation_sheet.xlsx
│   ├── annotation_sheet_01.xlsx 
│   └── annotation_sheet_02.xlsx
│
└── source/
    ├── build_simp.py           # merge original, reference and LLM simplification
    ├── llm_simplifying.py      # LLM simplification
    ├── simp_eval.py            # automatic simplification metric evaluation
    ├── human_eval.py           # human evaluation results analysis
    ├── human_eval_sample.py    # human evaluation data sampling = 60 samples
    ├── dating.py               # dating
    ├── date_eval.py            # dating results evaluation
    ├── ngram_trend_simple.py   # diarchronic linguistic analysis, mapping ADD + DELETE + KEEP tokens to monitor corpus
    ├── preprocessing.py        # sampling + transfer to jsonl
    └── prompts.py              # prompts template = dating + BASE simplification + UPTODATE simplification
```


---



## 3. Installation
### 3.1 Create Environment
```bash
python -m venv simp_env
source simp_env/bin/activate
```

### 3.2 Install Dependencies
```bash
pip install -r requirements.txt
```
If you encounter issues with EASSE:
```bash
git clone https://github.com/feralvam/easse.git
cd easse
pip install -e .
```



---



## 4. Reproducing the Experiments
### 4.0 preprocessing
--sample_size = 600, --seed = 42
```bash
python source/preprocessing.py
```
### 4.1 LLM-based Simplification
First use LLM to simplify the text, default : --model = gpt-5.2 , --temperature = 0.3, --sleep = 0.1
```bash
python source/llm_simplifying.py
```
then merge all sentences:
```bash
python source/build_simp.py
```


### 4.2 LLM-based Document Dating
Estimate temporal span (2020-2025): default parameters: --model = gpt-5.2 , --temperature = 0.3, --sleep = 0.1 (API rate limit control)
```bash
python source/dating.py data/tune_sample_simp.jsonl
```
Then evaluate centered predicted year: default: --target_year = 2025.0, --num_refs = 8, --sanity_n = 0, --p_threshold = 0.001
```bash
python source/date_eval.py data/tune_sample_dated.jsonl
```


### 4.3 Diachronic Linguisitc Analysis
```bash
python source/ngram_trend_simple.py
```


### 4.4 Automatic Simplification Evaluation 
default: --num_refs=8, --sari_tokenizer=13a, --bertscore_model=roberta-large, --bertscore_batch_size=32
```bash
python source/simp_eval.py data/tune_sample_simp.jsonl
```

### 4.5 Human Evaluation
First, sampling annotation data, default = --n_samples = 60, --seed = 42
```bash
python source/human_eval_sample.py
```
Then, evaluation:
```bash
python source/human_eval.py --csv1 data/annotation_sheet_01.csv --csv2 data/annotation_sheet_02.csv
```


## Data Source
The simplification dataset is based on the TurkCorpus dataset: 
Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen, and Chris Callison-Burch. 2016. Optimizing Statistical Machine Translation for Text Simplification. Transactions of the Association for Computational Linguistics, 4:401–415.