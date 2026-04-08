import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from path_utils import (
    ensure_dir,
    get_data_dir,
    get_processing_output_dir,
    get_project_root,
    relative_path,
)


LABEL_MAP = {"pos": 1, "neg": 0}
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d+")
WORD_STEM_RE = re.compile(r"[a-z']+")
TOKEN_RE = re.compile(r"[A-Za-z']+")
STEMMER = PorterStemmer()

NEGATION_WORDS = [
    "not",
    "no",
    "never",
    "none",
    "nor",
    "cannot",
    "can't",
    "won't",
    "don't",
    "didn't",
    "isn't",
    "wasn't",
]
POSITIVE_WORDS = [
    "good",
    "great",
    "amazing",
    "excellent",
    "fantastic",
    "love",
    "wonderful",
    "best",
]
NEGATIVE_WORDS = [
    "bad",
    "awful",
    "terrible",
    "horrible",
    "hate",
    "poor",
    "worst",
    "boring",
]
POSITIVE_WORDS_STEMMED = [STEMMER.stem(word) for word in POSITIVE_WORDS]
NEGATIVE_WORDS_STEMMED = [STEMMER.stem(word) for word in NEGATIVE_WORDS]


def load_reviews(data_dir) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    get_project_root()  # ensures path utils initialized
    review_id = 0
    for split in ("train", "test"):
        for label_name, label_val in LABEL_MAP.items():
            split_dir = data_dir / split / label_name
            if not split_dir.exists():
                raise FileNotFoundError(f"Missing directory: {split_dir}")
            for file_path in sorted(split_dir.glob("*.txt")):
                text = file_path.read_text(encoding="utf-8", errors="replace")
                records.append(
                    {
                        "review_id": review_id,
                        "split": split,
                        "label": label_val,
                        "label_name": label_name,
                        "file_path": relative_path(file_path),
                        "raw_review": text,
                    }
                )
                review_id += 1
    return pd.DataFrame(records)


def apply_stemming(text: str) -> str:
    def replace(match: re.Match) -> str:
        return STEMMER.stem(match.group(0))

    return WORD_STEM_RE.sub(replace, text)


def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = text.lower()
    text = DIGIT_RE.sub(" ", text)
    text = apply_stemming(text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_reviews(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df["cleaned_review"] = df["raw_review"].fillna("").apply(clean_text)
    total_raw_removed = 0
    total_clean_removed = 0
    per_split_stats = {}
    frames = []
    for split, group in df.groupby("split"):
        before = len(group)
        dedup_raw = group.drop_duplicates(subset="raw_review")
        removed_raw = before - len(dedup_raw)
        dedup_clean = dedup_raw.drop_duplicates(subset="cleaned_review")
        removed_clean = len(dedup_raw) - len(dedup_clean)
        total_raw_removed += removed_raw
        total_clean_removed += removed_clean
        per_split_stats[split] = {
            "initial": before,
            "after_raw_dedup": len(dedup_raw),
            "after_clean_dedup": len(dedup_clean),
            "removed_raw_duplicates": removed_raw,
            "removed_cleaned_duplicates": removed_clean,
        }
        frames.append(dedup_clean)
    dedup_df = (
        pd.concat(frames, ignore_index=False)
        .sort_values("review_id")
        .reset_index(drop=True)
    )
    stats = {
        "initial_rows": int(len(df)),
        "removed_exact_raw_duplicates": int(total_raw_removed),
        "removed_cleaned_duplicates": int(total_clean_removed),
        "final_rows": int(len(dedup_df)),
        "per_split": per_split_stats,
    }
    return dedup_df, stats


def save_cleaned_data(df: pd.DataFrame, output_dir) -> Tuple[str, str]:
    ensure_dir(output_dir)
    cleaned_path = output_dir / "cleaned_reviews.csv"
    labels_path = output_dir / "labels.csv"
    df.to_csv(
        cleaned_path,
        columns=[
            "review_id",
            "split",
            "label",
            "label_name",
            "file_path",
            "raw_review",
            "cleaned_review",
        ],
        index=False,
    )
    df[["review_id", "split", "label", "label_name"]].to_csv(labels_path, index=False)
    return str(cleaned_path), str(labels_path)


def build_tfidf_features(
    df: pd.DataFrame,
    vectorizer_params: Dict[str, object],
    train_matrix_filename: str,
    test_matrix_filename: str,
    feature_names_filename: str,
    vectorizer_filename: str,
    output_dir,
) -> Dict[str, object]:
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    train_texts = df.loc[train_mask, "cleaned_review"].tolist()
    test_texts = df.loc[test_mask, "cleaned_review"].tolist()

    vectorizer = TfidfVectorizer(**vectorizer_params)
    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)

    train_matrix_path = output_dir / train_matrix_filename
    test_matrix_path = output_dir / test_matrix_filename
    sparse.save_npz(train_matrix_path, train_matrix)
    sparse.save_npz(test_matrix_path, test_matrix)

    feature_names = vectorizer.get_feature_names_out().tolist()
    (output_dir / feature_names_filename).write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dump(vectorizer, output_dir / vectorizer_filename)

    return {
        "train_shape": (int(train_matrix.shape[0]), int(train_matrix.shape[1])),
        "test_shape": (int(test_matrix.shape[0]), int(test_matrix.shape[1])),
        "vectorizer_params": vectorizer_params,
        "feature_names_path": str(output_dir / feature_names_filename),
        "train_matrix_path": str(train_matrix_path),
        "test_matrix_path": str(test_matrix_path),
    }


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    negation_set = set(NEGATION_WORDS)
    positive_set = set(POSITIVE_WORDS_STEMMED)
    negative_set = set(NEGATIVE_WORDS_STEMMED)

    rows = []
    for _, row in df.iterrows():
        text = row["cleaned_review"]
        words = TOKEN_RE.findall(text)
        char_count = len(text)
        word_count = len(words)
        avg_word_length = float(sum(len(w) for w in words) / word_count) if word_count else 0.0
        token_counter = Counter(words)
        negation_count = sum(token_counter.get(token, 0) for token in negation_set)
        positive_count = sum(token_counter.get(token, 0) for token in positive_set)
        negative_count = sum(token_counter.get(token, 0) for token in negative_set)
        rows.append(
            {
                "review_id": row["review_id"],
                "split": row["split"],
                "label": row["label"],
                "label_name": row["label_name"],
                "char_count": char_count,
                "word_count": word_count,
                "avg_word_length": avg_word_length,
                "exclaim_count": text.count("!"),
                "question_count": text.count("?"),
                "negation_count": negation_count,
                "positive_word_count": positive_count,
                "negative_word_count": negative_count,
            }
        )
    return pd.DataFrame(rows)


def write_feature_readme(output_dir) -> None:
    readme_path = output_dir / "feature_readme.md"
    content = """# Feature Engineering Deliverables

## Preprocessing Summary
- HTML tags removed, text lowercased, digits stripped, and whitespace normalized
- Porter stemming applied while preserving negation words and emotive punctuation
- Duplicates removed within each split to avoid cross-split leakage
- Vectorizers are trained strictly on training data; test reviews are only transformed

## Output Files
- `cleaned_reviews.csv`: raw and cleaned text alongside identifiers
- `labels.csv`: minimal label table for aligning predictions
- `features_v1_train_tfidf_unigram.npz` / `features_v1_test_tfidf_unigram.npz`: unigram TF-IDF matrices with shared `v1_vectorizer.joblib` and feature names JSON
- `features_v2_train_tfidf_uni_bigram.npz` / `features_v2_test_tfidf_uni_bigram.npz`: unigram+bigram TF-IDF matrices with shared vectorizer artifacts
- `features_v3_statistical.csv`: interpretable numeric features
- `preprocessing_log.json`: detailed preprocessing and parameter log
- `feature_shapes.json`: matrix dimension metadata

## Version Guidelines
- **V1 (Unigram TF-IDF)**: suited for Logistic Regression, Naive Bayes, and Linear SVM
- **V2 (Uni+Bigram TF-IDF)**: suited for Logistic Regression and Linear SVM where phrase cues matter
- **V3 (Statistical Features)**: suited for Decision Trees, Random Forests, and other interpretable models

## Alignment Instructions
Rows in all artifacts maintain the order of `cleaned_reviews.csv`. Load the appropriate train/test TF-IDF files with the provided vectorizer to keep evaluation consistent, and align predictions via `review_id` and `labels.csv`.
"""
    readme_path.write_text(content.strip() + "\n", encoding="utf-8")


def write_preprocessing_log(output_dir, preprocess_stats, v1_info, v2_info) -> None:
    log_path = output_dir / "preprocessing_log.json"
    log_data = {
        "preprocessing": {
            "remove_html": True,
            "lowercase": True,
            "remove_digits": True,
            "normalize_whitespace": True,
            "stemming": True,
            "stemmer": "PorterStemmer",
            "deduplicate_per_split": True,
            "keep_negation": True,
        },
        "deduplication_stats": preprocess_stats,
        "v1_vectorizer_params": {
            **v1_info["vectorizer_params"],
            "fit_split": "train_only",
        },
        "v2_vectorizer_params": {
            **v2_info["vectorizer_params"],
            "fit_split": "train_only",
        },
        "v3_features": {
            "char_count": "len(cleaned_review)",
            "word_count": "token count (alphabetic)",
            "avg_word_length": "mean token length",
            "exclaim_count": "count of '!'",
            "question_count": "count of '?'",
            "negation_count": NEGATION_WORDS,
            "positive_word_count": {
                "lexicon_words": POSITIVE_WORDS,
                "stemmed_tokens": POSITIVE_WORDS_STEMMED,
            },
            "negative_word_count": {
                "lexicon_words": NEGATIVE_WORDS,
                "stemmed_tokens": NEGATIVE_WORDS_STEMMED,
            },
        },
    }
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_feature_shapes(
    output_dir,
    df_len: int,
    labels_len: int,
    v1_train_shape: Tuple[int, int],
    v1_test_shape: Tuple[int, int],
    v2_train_shape: Tuple[int, int],
    v2_test_shape: Tuple[int, int],
    v3_shape: Tuple[int, int],
) -> None:
    shapes_path = output_dir / "feature_shapes.json"
    data = {
        "cleaned_reviews_rows": df_len,
        "labels_rows": labels_len,
        "v1_train_shape": list(v1_train_shape),
        "v1_test_shape": list(v1_test_shape),
        "v2_train_shape": list(v2_train_shape),
        "v2_test_shape": list(v2_test_shape),
        "v3_shape": list(v3_shape),
    }
    shapes_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    data_dir = get_data_dir()
    output_dir = ensure_dir(get_processing_output_dir())
    df = load_reviews(data_dir)
    df, preprocess_stats = preprocess_reviews(df)
    cleaned_path, labels_path = save_cleaned_data(df, output_dir)

    v1_params = {
        "ngram_range": (1, 1),
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.95,
    }
    v2_params = {
        "ngram_range": (1, 2),
        "max_features": 8000,
        "min_df": 3,
        "max_df": 0.95,
    }

    v1_info = build_tfidf_features(
        df,
        v1_params,
        "features_v1_train_tfidf_unigram.npz",
        "features_v1_test_tfidf_unigram.npz",
        "features_v1_feature_names.json",
        "v1_vectorizer.joblib",
        output_dir,
    )
    v2_info = build_tfidf_features(
        df,
        v2_params,
        "features_v2_train_tfidf_uni_bigram.npz",
        "features_v2_test_tfidf_uni_bigram.npz",
        "features_v2_feature_names.json",
        "v2_vectorizer.joblib",
        output_dir,
    )

    v3_df = create_statistical_features(df)
    v3_path = output_dir / "features_v3_statistical.csv"
    v3_df.to_csv(v3_path, index=False)

    write_feature_readme(output_dir)
    write_preprocessing_log(output_dir, preprocess_stats, v1_info, v2_info)
    write_feature_shapes(
        output_dir,
        len(df),
        len(df),
        v1_info["train_shape"],
        v1_info["test_shape"],
        v2_info["train_shape"],
        v2_info["test_shape"],
        (len(v3_df), v3_df.shape[1]),
    )

    print("Preprocessing complete.")
    print(f"Cleaned reviews: {cleaned_path}")
    print(f"Labels: {labels_path}")
    print(f"V1 train shape: {v1_info['train_shape']}, test shape: {v1_info['test_shape']}")
    print(f"V2 train shape: {v2_info['train_shape']}, test shape: {v2_info['test_shape']}")
    print(f"V3 features saved to: {v3_path}")


if __name__ == "__main__":
    main()
