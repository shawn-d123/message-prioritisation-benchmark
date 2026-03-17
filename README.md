This is a professional, structured version of your README. It uses clear hierarchy, visual elements, and concise language to highlight both your technical skills and your analytical process—which is key for a portfolio piece.

-----

# Message Prioritisation Benchmark

A lightweight NLP benchmark for classifying short email-style messages by **operational priority** and **action requirement**. This project compares an interpretable **rule-based baseline** against classical **scikit-learn** models to determine the most effective triage strategy for high-volume inboxes.

-----

## 📊 Interactive Visual Summary

The chart below visualizes the performance delta between the rule-based approach and machine learning models.

[](https://www.google.com/search?q=%5Bhttps://public.flourish.studio/visualisation/28101508/%5D\(https://public.flourish.studio/visualisation/28101508/\))

*Interactive version: [View the Flourish chart](https://public.flourish.studio/visualisation/28101508/)*

-----

## 🎯 Project Overview

This project explores a common workplace bottleneck: **Message Triage**. Not every message requires the same response. This benchmark tests whether lightweight NLP can accurately categorize messages into two distinct dimensions:

### 1\. Priority Classification

  * `Urgent`: Immediate attention required.
  * `Important`: High-value, but not time-critical.
  * `Routine`: Standard operational tasks.
  * `Informational`: No immediate response needed.

### 2\. Action Detection

  * `Action Required`: **True** or **False**.

### The Contenders

I compared three distinct approaches:

  * **Rule-based Baseline:** Hand-crafted keyword patterns and heuristics.
  * **TF-IDF + Multinomial Naive Bayes:** Probabilistic classifier focusing on word frequencies.
  * **TF-IDF + Logistic Regression:** Linear model for high-dimensional text data.

-----

## 🧠 Why I Built This

This project demonstrates an end-to-end data science workflow, specifically focusing on **decision support**. It highlights my ability to:

  * **Design NLP Benchmarks:** Designing taxonomies and label structures.
  * **Curation:** Building and validating a custom dataset.
  * **Comparative Analysis:** Moving beyond "black box" ML by testing against a transparent baseline.
  * **Failure Analysis:** Identifying exactly where and why models struggle.

-----

## 📈 Dataset Evolution

A critical finding in this project was how **dataset maturity** impacts model selection.

| Phase | Size | Observation |
| :--- | :--- | :--- |
| **Initial** | Small/Unbalanced | Rule-based baseline outperformed ML; high instability. |
| **Final** | **240 rows** | Balanced classes (60 each) allowed ML models to stabilize and surpass the baseline. |

### Final Composition

  * **Total Rows:** 240
  * **Distribution:** 60 per priority class (Urgent, Important, Routine, Informational).
  * **Action Split:** 160 True / 80 False.
  * **Split:** 75% Train (180) / 25% Test (60).

-----

## 🏆 Final Results

On the final balanced benchmark, **Multinomial Naive Bayes** emerged as the superior choice for short-text triage.

### Model Performance

| Model | Priority Accuracy | Priority Macro F1 | Action Required F1 |
| :--- | :---: | :---: | :---: |
| Rule Baseline | 0.4333 | 0.4036 | 0.6842 |
| **Naive Bayes** | **0.5500** | **0.5407** | **0.7835** |
| Logistic Regression | 0.4833 | 0.4669 | 0.7755 |

### Key Takeaways

1.  **ML Scales Better:** While rule-based methods are great for cold-starts, classical ML (specifically Naive Bayes) becomes significantly more reliable once a balanced dataset is available.
2.  **Binary is Easier:** All models performed better at binary `action_required` detection than 4-class priority classification.
3.  **Naive Bayes for Short Text:** The model's reliance on keyword signals makes it highly effective for the sparse features found in short email-style messages.

-----

## 🛠️ Project Structure & Workflow

The project is modularized into a pipeline for reproducibility:

```text
├── data/               # Raw and processed CSVs
├── outputs/            # Predictions, metrics, and matplotlib charts
├── src/
│   ├── build_dataset.py       # Standardizes raw data
│   ├── validate_dataset.py    # Schema and integrity checks
│   ├── split_dataset.py       # Reproducible stratified splitting
│   ├── rule_priority_baseline.py # Keyword-based logic
│   ├── train_models.py        # scikit-learn training pipeline
│   ├── evaluate_models.py     # Accuracy/F1 calculation
│   ├── analyse_errors.py      # Confusion matrix & failure rows
│   └── generate_visualisations.py # Chart generation
└── requirements.txt
```

### How to Run

```bash
# Execute the full pipeline in sequence
python src/build_dataset.py && \
python src/validate_dataset.py && \
python src/split_dataset.py && \
python src/rule_priority_baseline.py && \
python src/train_models.py && \
python src/evaluate_models.py && \
python src/analyse_errors.py && \
python src/generate_visualisations.py
```

-----

## 🚀 Future Roadmap

  * **Real-world Integration:** Incorporating non-synthetic message sources.
  * **Advanced Models:** Testing Support Vector Machines (SVM) and Cross-Validation.
  * **LLM Comparison:** Testing local, lightweight LLM classifiers (e.g., Llama 3/Mistral) against these classical methods.
  * **Confidence Scores:** Introducing a "Uncertain" flag for low-probability predictions.

-----

## 👤 Author

**Shawn Santan D'Souza**

  * **GitHub:** [shawn-d123](https://www.google.com/search?q=https://github.com/shawn-d123)
  * **LinkedIn:** [Shawn D'Souza](https://www.google.com/search?q=https://www.linkedin.com/in/shawn-dsouza-code234)

*License: [MIT](https://www.google.com/search?q=LICENSE)*

-----

**Would you like me to add a "Visualizations" section with placeholders for the `.png` files generated by your script?**
