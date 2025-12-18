# Financial News NLP — Narrative & Topic Analysis

This project applies Natural Language Processing (NLP) to financial news headlines from **CNBC, Reuters, and The Guardian** to identify dominant market narratives and compare how different outlets frame economic events.

Rather than predicting prices, the goal is **interpretability**: understanding *what stories are being told*, *which themes dominate*, and *how narrative emphasis varies across sources*.

---

## 📌 Project Objectives

- Clean and normalize raw financial news text
- Establish an interpretable NLP baseline (TF-IDF)
- Discover recurring themes via clustering and topic modeling
- Compare narrative distributions across news outlets
- Produce human-readable insights suitable for analysis and reporting

---

## 📊 Datasets

- **CNBC Headlines**
- **Reuters Headlines**
- **The Guardian Headlines**

Each record represents a single published headline with associated metadata (time, source).

---

## 🧠 Methodology

### 1. Exploratory Data Analysis & Problem Framing
- Inspect dataset structure and coverage
- Define headlines as the primary NLP input
- Frame the task as **unsupervised narrative discovery**

### 2. Text Cleaning & Normalization
- Lowercasing, whitespace normalization
- Conservative punctuation handling (finance-aware)
- Deduplication and missing-value handling
- Persist cleaned data for reproducibility

### 3. TF-IDF Baseline + Clustering
- Convert text to TF-IDF features
- Apply KMeans clustering
- Interpret clusters using top-weighted terms
- Compare cluster prevalence by source

### 4. Topic Modeling (LDA)
- Apply Latent Dirichlet Allocation (LDA)
- Extract dominant narrative topics
- Assign topics to individual headlines
- Analyze topic distribution across outlets

### 5. Final Synthesis
- Identify the most interpretable and recurring narratives
- Highlight framing differences between CNBC, Reuters, and The Guardian
- Discuss limitations and potential extensions

---

## 📁 Repository Structure
```
financial-news-nlp/
├── data/
│ ├── raw/ # Original headline CSVs
│ └── processed/ # Cleaned & modeled outputs
├── notebooks/
│ ├── 01_eda_problem_framing.ipynb
│ ├── 02_text_cleaning_normalization.ipynb
│ ├── 03_tfidf_clustering_baseline.ipynb
│ ├── 04_topic_modeling_narratives.ipynb
│ └── 05_final_summary.ipynb
├── README.md
```

---

## 🔍 Key Findings (High-Level)

- A small number of themes dominate financial coverage (markets, earnings, macro policy, geopolitics)
- Reuters tends to emphasize factual, macro-driven narratives
- CNBC leans toward market reaction and investor sentiment
- The Guardian provides broader socio-economic framing
- Topic modeling provides clearer narrative separation than clustering alone

---

## ⚠️ Limitations

- Headlines only (no full article bodies)
- Unsupervised methods may mix adjacent themes
- Topic counts are heuristic, not ground truth

---

## 🚀 Next Steps

- Replace bag-of-words with transformer embeddings
- Track narrative evolution over time
- Extend analysis to full articles or social media
- Use narrative signals as features in downstream financial models

---

## 🛠️ Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- Matplotlib
- Jupyter

---

## ✍🏽 Author

Brandon Theard  
Data Scientist | NLP & Financial Analysis