import os
import io
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


st.set_page_config(page_title="Financial News NLP Dashboard", page_icon="🗞️", layout="wide")

# repo-root/data/raw/*.csv
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))


def standardize_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Headlines" not in df.columns:
        raise ValueError(f"{source_name}: expected column 'Headlines' not found. Found: {list(df.columns)}")

    df = df.rename(columns={"Headlines": "headline", "Time": "date", "Description": "description"})

    if "date" not in df.columns:
        df["date"] = ""
    if "description" not in df.columns:
        df["description"] = ""

    df["source"] = source_name

    df["headline"] = (
        df["headline"].astype(str).fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df[df["headline"].str.len() > 0].drop_duplicates(subset=["headline", "source"]).reset_index(drop=True)

    return df[["headline", "source", "date", "description"]]


def load_repo_raw_data() -> pd.DataFrame:
    files = {
        "CNBC": os.path.join(RAW_DIR, "cnbc_headlines.csv"),
        "Guardian": os.path.join(RAW_DIR, "guardian_headlines.csv"),
        "Reuters": os.path.join(RAW_DIR, "reuters_headlines.csv"),
    }

    frames = []
    for src, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}. Expected it under data/raw/")
        raw = pd.read_csv(path)
        frames.append(standardize_columns(raw, src))

    return pd.concat(frames, ignore_index=True)


def read_uploaded_csvs(uploads: List) -> pd.DataFrame:
    frames = []
    for up in uploads:
        name = up.name.lower()
        raw = pd.read_csv(io.BytesIO(up.getvalue()))

        if "cnbc" in name:
            src = "CNBC"
        elif "guardian" in name:
            src = "Guardian"
        elif "reuters" in name:
            src = "Reuters"
        else:
            src = "Unknown"

        frames.append(standardize_columns(raw, src))

    return pd.concat(frames, ignore_index=True)


def top_terms_from_components(
    components: np.ndarray, feature_names: List[str], top_n: int
) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for i, weights in enumerate(components):
        idx = np.argsort(weights)[::-1][:top_n]
        out[i] = [feature_names[j] for j in idx]
    return out


@st.cache_data(show_spinner=False)
def run_models(df: pd.DataFrame, n_clusters: int, n_topics: int, max_features: int, top_n_terms: int):
    # TF-IDF + KMeans
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2 if len(df) >= 100 else 1,
    )
    X = tfidf.fit_transform(df["headline"].tolist())

    k = max(2, min(n_clusters, max(2, len(df) - 1)))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    clusters = km.fit_predict(X)

    df_out = df.copy()
    df_out["cluster"] = clusters

    feature_names = tfidf.get_feature_names_out().tolist()
    centroids = km.cluster_centers_
    cluster_terms = {}
    for ci in range(centroids.shape[0]):
        idx = np.argsort(centroids[ci])[::-1][:top_n_terms]
        cluster_terms[ci] = [feature_names[j] for j in idx]

    # LDA Topics
    cv = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2 if len(df) >= 100 else 1,
    )
    Xc = cv.fit_transform(df["headline"].tolist())

    t = max(2, min(n_topics, max(2, len(df) - 1)))
    lda = LatentDirichletAllocation(
        n_components=t,
        random_state=42,
        learning_method="batch",
        max_iter=10,
    )
    doc_topic = lda.fit_transform(Xc)

    df_out["topic"] = doc_topic.argmax(axis=1)
    df_out["topic_conf"] = doc_topic.max(axis=1)

    topic_terms = top_terms_from_components(
        lda.components_,
        cv.get_feature_names_out().tolist(),
        top_n_terms
    )

    return df_out, cluster_terms, topic_terms


def main():
    st.title("🗞️ Financial News NLP — Interactive Dashboard")
    st.caption("Compare **narratives (clusters)** and **themes (topics)** across CNBC, Guardian, and Reuters.")

    # -------- Sidebar (DO NOT reference df here yet) --------
    with st.sidebar:
        st.header("Data")
        uploads = st.file_uploader(
            "Upload 1–3 CSVs (optional)",
            type=["csv"],
            accept_multiple_files=True,
            help="If you upload files, we auto-label source by filename (cnbc/guardian/reuters).",
        )
        use_repo_data = st.toggle("Use repo datasets (data/raw)", value=(len(uploads) == 0))
        st.caption(f"Expected repo path: {RAW_DIR}")

        st.divider()
        st.header("Model settings")

        # A) Sampling slider (safe upper bound; we clamp after loading df)
        sample_size = st.slider(
            "Sample size (for speed)",
            min_value=2000,
            max_value=60000,   # safe cap; we clamp to len(df) later
            value=6000,
            step=1000,
        )

        n_clusters = st.slider("Clusters (KMeans)", 2, 12, 6)
        n_topics = st.slider("Topics (LDA)", 2, 12, 6)
        max_features = st.slider("Max features", 500, 10000, 3000, step=500)
        top_n_terms = st.slider("Top terms shown", 5, 20, 10)

        # B) Run button so the app doesn't retrain on every slider wiggle
        st.divider()
        run = st.button("Run analysis")

    # -------- Load data --------
    try:
        if use_repo_data:
            df = load_repo_raw_data()
        else:
            if len(uploads) == 0:
                st.warning("Upload CSVs or toggle 'Use repo datasets' on.")
                st.stop()
            df = read_uploaded_csvs(uploads)
    except Exception as e:
        st.error(f"Data load error: {e}")
        st.stop()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headlines", f"{len(df):,}")
    c2.metric("Sources", f"{df['source'].nunique():,}")
    c3.metric("Has dates", "Yes" if df["date"].astype(str).str.len().sum() > 0 else "No")
    c4.metric("Has descriptions", "Yes" if df["description"].astype(str).str.len().sum() > 0 else "No")

    st.divider()

    # B) Require click to run (feels like a real app)
    if not run:
        st.info("Adjust settings on the left, then click **Run analysis**.")
        st.stop()

    # A) Clamp sample size now that df exists
    effective_sample = int(min(sample_size, len(df)))

    with st.spinner(f"Training models on {effective_sample:,} headlines…"):
        df_model = df.sample(n=effective_sample, random_state=42) if effective_sample < len(df) else df
        df_out, cluster_terms, topic_terms = run_models(df_model, n_clusters, n_topics, max_features, top_n_terms)

    # -------- Charts --------
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Narrative clusters (TF-IDF + KMeans)")
        st.write("Clusters group headlines with similar phrasing — **recurring storylines**.")
        counts = df_out.groupby(["cluster", "source"]).size().reset_index(name="count")
        st.bar_chart(counts.pivot_table(index="cluster", columns="source", values="count", fill_value=0))

        st.markdown("#### Cluster labels (top terms)")
        labels = pd.DataFrame({
            "cluster": list(cluster_terms.keys()),
            "top_terms": [", ".join(v) for v in cluster_terms.values()],
        }).sort_values("cluster")
        st.dataframe(labels, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Themes (LDA Topic Model)")
        st.write("Topics are broader themes — a softer grouping than clusters.")
        tcounts = df_out.groupby(["topic", "source"]).size().reset_index(name="count")
        st.bar_chart(tcounts.pivot_table(index="topic", columns="source", values="count", fill_value=0))

        st.markdown("#### Topic labels (top terms)")
        tlabels = pd.DataFrame({
            "topic": list(topic_terms.keys()),
            "top_terms": [", ".join(v) for v in topic_terms.values()],
        }).sort_values("topic")
        st.dataframe(tlabels, use_container_width=True, hide_index=True)

    # -------- Explorer --------
    st.divider()
    st.subheader("Explorer")

    a, b, c = st.columns([0.5, 0.25, 0.25])
    with a:
        q = st.text_input("Search headlines", placeholder="e.g., inflation, earnings, oil, layoffs")
    with b:
        src = st.multiselect("Source", sorted(df_out["source"].unique()), default=sorted(df_out["source"].unique()))
    with c:
        topic = st.multiselect("Topic", sorted(df_out["topic"].unique()), default=sorted(df_out["topic"].unique()))

    view = df_out[df_out["source"].isin(src) & df_out["topic"].isin(topic)].copy()
    if q and q.strip():
        view = view[view["headline"].str.contains(q.strip(), case=False, na=False)]

    st.dataframe(
        view[["date", "source", "headline", "cluster", "topic", "topic_conf"]]
        .sort_values(["source", "topic_conf"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )

    # -------- Export --------
    st.divider()
    st.subheader("Export")
    export = view[["date", "source", "headline", "cluster", "topic", "topic_conf"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download current view (CSV)", export, "nlp_dashboard_export.csv", "text/csv")


if __name__ == "__main__":
    main()
