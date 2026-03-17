<div align="center">

# 📬 Message Prioritisation Benchmark

**A lightweight NLP benchmark for classifying messages by priority and action requirement — comparing rule-based heuristics against classical ML.**

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-NLP_Benchmark-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557C?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

Not every message deserves the same attention. This benchmark tests whether lightweight NLP can support realistic message triage.

---

[![Model Performance Comparison](https://public.flourish.studio/visualisation/28101508/thumbnail)](https://public.flourish.studio/visualisation/28101508/)

**[View interactive chart on Flourish](https://public.flourish.studio/visualisation/28101508/)**

---

</div>

## About

This project builds a complete NLP benchmark pipeline for classifying short email-style messages across two tasks: **4-class priority classification** (urgent, important, routine, informational) and **binary action detection** (action required or not). Three approaches are compared on the same test set — a keyword-driven rule baseline, TF-IDF + Multinomial Naive Bayes, and TF-IDF + Logistic Regression.

The project was built to explore a practical workflow problem and to demonstrate end-to-end benchmark design: dataset curation, label taxonomy, train/test splitting, model training, evaluation, error analysis, and visualisation.

<br>

## Dataset

The final benchmark contains **240 labelled messages**, iteratively expanded and balanced after early experiments showed that a smaller dataset produced unstable results.

| | |
|:---|:---|
| **Total rows** | 240 (60 per priority class) |
| **Action split** | 160 true / 80 false |
| **Train / test** | 180 / 60 (15 test examples per class) |

<br>

## Methods

| Model | Approach | Notes |
|:---|:---|:---|
| **Rule baseline** | Keyword patterns for urgency, deadlines, review requests, informational cues | Interpretable and fast, but weaker at scale |
| **Multinomial Naive Bayes** | TF-IDF features, unigrams + bigrams | Strongest overall on the final benchmark |
| **Logistic Regression** | Same TF-IDF representation | Improved over the rule baseline, but did not beat Naive Bayes |

<br>

## Results

### Overall comparison

| Model | Priority Accuracy | Priority Macro F1 | Priority Weighted F1 | Action Required F1 |
|:---|---:|---:|---:|---:|
| Rule Baseline | 0.4333 | 0.4036 | 0.4036 | 0.6842 |
| **Naive Bayes** | **0.5500** | **0.5407** | **0.5407** | **0.7835** |
| Logistic Regression | 0.4833 | 0.4669 | 0.4669 | 0.7755 |

### Error summary

| Model | Total Errors | Priority Errors | Action Errors |
|:---|---:|---:|---:|
| Rule Baseline | 41 | 34 | 24 |
| **Naive Bayes** | **36** | **27** | **21** |
| Logistic Regression | 41 | 31 | 22 |

<br>

## Key Findings

**Dataset design changed the outcome.** On a smaller benchmark, the rule baseline appeared stronger. Once the dataset was expanded and balanced to 240 rows, classical ML clearly improved and Naive Bayes emerged as the best method. This was the single most important finding from the project.

**Naive Bayes fit short text triage well.** The task relies on keyword patterns, short phrase signals, and compact message structure — a natural match for Multinomial Naive Bayes with TF-IDF features.

**Action detection was easier than priority classification.** All models scored higher on binary action detection than on 4-class priority, suggesting that "needs action or not" is a simpler signal to learn.

**Interpretable baselines still matter.** The rule baseline was weaker overall but provided a transparent benchmark, showed where hand-written heuristics break down, and helped explain what the ML models were improving on.

<br>

## Pipeline

```
1. build_dataset.py         → Load and standardise raw data
2. validate_dataset.py      → Check schema, labels, duplicates, empty text
3. split_dataset.py         → Stratified train/test split
4. rule_priority_baseline.py → Run keyword-based baseline
5. train_models.py          → Train Naive Bayes and Logistic Regression
6. evaluate_models.py       → Accuracy, macro F1, weighted F1, classification reports
7. analyse_errors.py        → Error summaries, confusion pairs, failure rows
8. generate_visualisations.py → Comparison charts
```

<br>

## Project Structure

```
message-prioritisation-benchmark/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   ├── analysis/
│   └── charts/
├── src/
│   ├── build_dataset.py
│   ├── schemas.py
│   ├── validate_dataset.py
│   ├── split_dataset.py
│   ├── rule_priority_baseline.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── analyse_errors.py
│   └── generate_visualisations.py
├── README.md
├── requirements.txt
└── .gitignore
```

<br>

## Tech Stack

| | |
|:---|:---|
| **Language** | Python |
| **Data** | Pandas, CSV-based pipeline |
| **ML** | scikit-learn (TF-IDF, Naive Bayes, Logistic Regression) |
| **Visualisation** | Matplotlib, Flourish |

<br>

## How to Run

```bash
python src/build_dataset.py
python src/validate_dataset.py
python src/split_dataset.py
python src/rule_priority_baseline.py
python src/train_models.py
python src/evaluate_models.py
python src/analyse_errors.py
python src/generate_visualisations.py
```

<br>

## What I Practised

- Designing an NLP benchmark from scratch
- Creating and validating a labelled dataset with balanced classes
- Comparing rule-based and classical ML approaches fairly
- Working with TF-IDF features and scikit-learn classifiers
- Evaluating with accuracy and F1 (macro and weighted)
- Investigating failures through structured error analysis
- Iterating on dataset design to improve reliability
- Communicating findings with tables, charts, and interactive visuals

<br>

## Future Improvements

| Area | Idea |
|:---|:---|
| Data | More real-world, non-synthetic message sources |
| Coverage | Subtle edge cases and ambiguous messages |
| Models | SVM, cross-validation, confidence scores |
| Comparison | Lightweight local LLM classifier |
| Interface | Simple dashboard or demo |

<br>

---

<div align="center">

**An end-to-end NLP benchmark demonstrating dataset design, classical ML evaluation, error analysis, and clear communication of findings.**

MIT License

[Shawn Santan D'Souza](https://github.com/shawn-d123) · [LinkedIn](https://www.linkedin.com/in/shawn-dsouza-code234)

</div>
