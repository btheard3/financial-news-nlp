# Financial News NLP — Narrative & Topic Analysis
🔗 Live App

http://alb-financial-news-nlp-27107617.us-east-2.elb.amazonaws.com/

## Problem

Markets move on stories, yet most financial news analysis collapses narrative into shallow sentiment scores. This ignores framing, emphasis, and timing—the factors that shape expectations and delayed market reactions.

## Why This Problem Matters

Narratives influence:

- Risk perception

- Volatility amplification

- Investor attention cycles

Understanding *how* stories are framed helps explain why markets react the way they do—even when the data is already known.

## Data Used

- Financial news headlines from:

    - **CNBC**

    - **Reuters**

    - **The Guardian**

- Fields include source, timestamp, and headline text

The focus is comparative framing, not proprietary data.

## Approach

- Text cleaning and normalization

- TF-IDF feature construction (interpretable baseline)

- K-Means clustering for headline grouping

- LDA topic modeling for dominant narratives

- Cross-source comparison of topic prevalence

## Evaluation & Findings

- Narrative emphasis differs significantly by outlet

- Consensus narratives form after price discovery

- Framing divergence explains delayed or exaggerated reactions

## Limitations

- Headline-only corpus

- Static topic modeling

- No direct price linkage yet

## Planned Next Steps

- Dynamic topic modeling to track narrative drift

- Event-aligned narrative analysis

- Full-article text ingestion

## Reproducibility — Run Locally
```bash
git clone https://github.com/btheard3/financial-news-nlp
cd financial-news-nlp
pip install -r requirements.txt
```


Run notebooks sequentially:
```
01_eda_problem_framing.ipynb
02_text_cleaning.ipynb
03_tfidf_modeling.ipynb
04_topic_modeling.ipynb
05_final_summary.ipynb
```

## Portfolio Context

**Narrative layer** — explains how market stories form around quantitative signals.

Author

Brandon Theard
Data Scientist | Decision-Support Systems

GitHub: https://github.com/btheard3

LinkedIn: https://www.linkedin.com/in/brandon-theard-811b38131/