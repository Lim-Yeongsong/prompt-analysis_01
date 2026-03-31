"""
Korean prompt text embedding — fully offline, no Java/HuggingFace required.

Method:
  - TF-IDF with character 2-3 gram + word unigram (handles Korean well)
  - TruncatedSVD → 100-dim dense vectors (L2-normalized)

Outputs:
  - embeddings_turn_level.csv          : 87 rows (one per turn) + metadata
  - embeddings_participant_level.csv   : 26 rows (per participant, turns joined) + metadata
  - embeddings_turn_level.npy          : shape (87, 100)
  - embeddings_participant_level.npy   : shape (26, 100)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline

N_COMPONENTS = 100
DATA_PATH = "prompt_260316.csv"


def make_pipeline(n_components: int) -> Pipeline:
    """TF-IDF (char 2-3 gram) + SVD pipeline."""
    tfidf = TfidfVectorizer(
        analyzer="char_wb",   # character n-gram with word boundaries
        ngram_range=(2, 3),   # bi/tri-char grams — captures Korean syllable structure
        min_df=1,
        sublinear_tf=True,
        strip_accents=None,
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return Pipeline([("tfidf", tfidf), ("svd", svd)])


def embed(texts, pipeline, fit: bool = True) -> np.ndarray:
    texts = [t if isinstance(t, str) else "" for t in texts]
    if fit:
        vecs = pipeline.fit_transform(texts)
    else:
        vecs = pipeline.transform(texts)
    return normalize(vecs, norm="l2")


def main():
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df)}, Participants: {df['participant_id'].nunique()}")

    # Fit pipeline on all turn texts so participant-level reuse is consistent
    all_texts = df["prompt_raw"].fillna("").tolist()
    n_components = min(N_COMPONENTS, len(all_texts) - 1)
    pipeline = make_pipeline(n_components)

    # ── 1. Turn-level (87 vectors) ─────────────────────────────────────────
    print(f"\n[1/2] Turn-level embeddings  (n={len(all_texts)}, dim={n_components})...")
    turn_embeddings = embed(all_texts, pipeline, fit=True)

    emb_cols = [f"emb_{i}" for i in range(turn_embeddings.shape[1])]
    df_turn = pd.concat(
        [df[["prompt_id", "turn", "participant_id", "geft_score"]].reset_index(drop=True),
         pd.DataFrame(turn_embeddings, columns=emb_cols)],
        axis=1,
    )
    df_turn.to_csv("embeddings_turn_level.csv", index=False)
    np.save("embeddings_turn_level.npy", turn_embeddings)
    print(f"  Saved: embeddings_turn_level.csv & .npy  {turn_embeddings.shape}")

    # ── 2. Participant-level (26 vectors) ──────────────────────────────────
    print(f"\n[2/2] Participant-level embeddings (turns concatenated)...")
    participant_df = (
        df.sort_values("turn")
          .groupby("participant_id", sort=False)
          .agg(geft_score=("geft_score", "first"),
               combined_text=("prompt_raw", lambda x: " ".join(x.fillna(""))))
          .reset_index()
    )

    participant_embeddings = embed(
        participant_df["combined_text"].tolist(), pipeline, fit=False
    )

    emb_cols_p = [f"emb_{i}" for i in range(participant_embeddings.shape[1])]
    df_participant = pd.concat(
        [participant_df[["participant_id", "geft_score"]].reset_index(drop=True),
         pd.DataFrame(participant_embeddings, columns=emb_cols_p)],
        axis=1,
    )
    df_participant.to_csv("embeddings_participant_level.csv", index=False)
    np.save("embeddings_participant_level.npy", participant_embeddings)
    print(f"  Saved: embeddings_participant_level.csv & .npy  {participant_embeddings.shape}")

    print("\nDone!")
    print(f"  Turn-level       : {turn_embeddings.shape}")
    print(f"  Participant-level: {participant_embeddings.shape}")


if __name__ == "__main__":
    main()
