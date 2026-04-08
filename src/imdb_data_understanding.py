import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from path_utils import (
    ensure_dir,
    get_data_dir,
    get_data_understanding_output_dir,
    relative_path,
)


LABEL_MAP = {"pos": 1, "neg": 0}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def load_reviews() -> pd.DataFrame:
    data_dir = get_data_dir()
    records = []
    for split in ("train", "test"):
        for label_name, label_val in LABEL_MAP.items():
            split_dir = data_dir / split / label_name
            if not split_dir.exists():
                raise FileNotFoundError(f"Missing directory: {split_dir}")
            for file_path in sorted(split_dir.glob("*.txt")):
                text = file_path.read_text(encoding="utf-8", errors="replace")
                records.append(
                    {
                        "split": split,
                        "label": label_val,
                        "label_name": label_name,
                        "file_path": relative_path(file_path),
                        "review": text,
                    }
                )
    return pd.DataFrame(records)


def dataset_overview(df: pd.DataFrame) -> dict:
    label_counts = df["label"].value_counts().to_dict()
    split_counts = df["split"].value_counts().to_dict()
    pos = int(label_counts.get(1, 0))
    neg = int(label_counts.get(0, 0))
    ratio = float(pos / neg) if neg else float("inf")
    review_series = df["review"]
    return {
        "total_samples": int(len(df)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "label_counts": {"positive": pos, "negative": neg},
        "class_balance_ratio_pos_over_neg": ratio,
        "missing_reviews": int(review_series.isna().sum()),
        "empty_reviews": int(review_series.fillna("").str.strip().eq("").sum()),
    }


def text_quality(df: pd.DataFrame) -> dict:
    review_text = df["review"].fillna("")
    html_pattern = re.compile(r"<[^>]+>")
    br_pattern = re.compile(r"<br\s*/?>", re.IGNORECASE)
    url_pattern = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
    repeated_punct_pattern = re.compile(r"(?:!{3,}|\?{3,}|\.{3,})")
    non_ascii = review_text.apply(lambda x: any(ord(ch) > 127 for ch in x))
    return {
        "html_tag_reviews": int(review_text.str.contains(html_pattern).sum()),
        "contains_br_reviews": int(review_text.str.contains(br_pattern).sum()),
        "url_reviews": int(review_text.str.contains(url_pattern).sum()),
        "non_ascii_reviews": int(non_ascii.sum()),
        "whitespace_only_reviews": int(review_text.str.strip().eq("").sum()),
        "repeated_punctuation_reviews": int(review_text.str.contains(repeated_punct_pattern).sum()),
    }


def add_length_features(df: pd.DataFrame) -> None:
    review_text = df["review"].fillna("")
    df["character_count"] = review_text.str.len().astype(int)
    df["word_count"] = review_text.apply(lambda x: len(x.split()))


def summarize_length(series: pd.Series) -> dict:
    values = series.to_numpy(dtype=float)
    if values.size == 0:
        return {k: 0.0 for k in ["min", "max", "mean", "median", "std", "pct_5", "pct_25", "pct_75", "pct_95"]}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "pct_5": float(np.percentile(values, 5)),
        "pct_25": float(np.percentile(values, 25)),
        "pct_75": float(np.percentile(values, 75)),
        "pct_95": float(np.percentile(values, 95)),
    }


def length_statistics(df: pd.DataFrame) -> dict:
    add_length_features(df)
    stats = {
        "overall": {
            "character_count": summarize_length(df["character_count"]),
            "word_count": summarize_length(df["word_count"]),
        },
        "by_split": {},
        "by_label": {},
    }
    for split, group in df.groupby("split"):
        stats["by_split"][split] = {
            "character_count": summarize_length(group["character_count"]),
            "word_count": summarize_length(group["word_count"]),
        }
    for label_name, group in df.groupby("label_name"):
        stats["by_label"][label_name] = {
            "character_count": summarize_length(group["character_count"]),
            "word_count": summarize_length(group["word_count"]),
        }
    return stats


def analyze_duplicates(df: pd.DataFrame) -> dict:
    duplicate_mask = df.duplicated(subset=["review"], keep=False)
    duplicates = df[duplicate_mask]
    if duplicates.empty:
        return {
            "duplicate_rows": 0,
            "duplicate_groups": 0,
            "conflicting_label_groups": 0,
            "conflicting_examples": [],
        }
    grouped = duplicates.groupby("review", sort=False)
    conflicting_examples = []
    conflict_count = 0
    for review_text, group in grouped:
        unique_labels = group["label"].unique()
        if len(unique_labels) > 1:
            conflict_count += 1
            snippet = re.sub(r"\s+", " ", review_text.strip())[:200]
            conflicting_examples.append(
                {
                    "snippet": snippet,
                    "labels": [int(x) for x in sorted(unique_labels)],
                    "file_paths": group["file_path"].tolist(),
                }
            )
            if len(conflicting_examples) >= 5:
                break
    return {
        "duplicate_rows": int(len(duplicates)),
        "duplicate_groups": int(grouped.ngroups),
        "conflicting_label_groups": conflict_count,
        "conflicting_examples": conflicting_examples,
    }


def vocabulary_analysis(df: pd.DataFrame):
    counter = Counter()
    for text in df["review"].fillna(""):
        counter.update(TOKEN_PATTERN.findall(text.lower()))
    total_tokens = int(sum(counter.values()))
    vocab_size = int(len(counter))
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stopwords = set(ENGLISH_STOP_WORDS)
    except Exception:
        stopwords = set()
    top_words = []
    for rank, (word, freq) in enumerate(counter.most_common(50), start=1):
        top_words.append(
            {
                "rank": rank,
                "word": word,
                "frequency": int(freq),
                "is_stopword": word in stopwords,
            }
        )
    return {
        "total_tokens": total_tokens,
        "vocabulary_size": vocab_size,
        "top_words": top_words,
        "counter": counter,
    }


def negation_counts(counter: Counter) -> dict:
    negations = ["not", "no", "never", "none", "nor", "cannot", "can't", "won't", "don't", "didn't", "isn't", "wasn't"]
    return {word: int(counter.get(word, 0)) for word in negations}


def sentiment_counts(counter: Counter) -> dict:
    positive = ["good", "great", "amazing", "excellent", "fantastic", "love", "wonderful", "best"]
    negative = ["bad", "awful", "terrible", "horrible", "hate", "poor", "worst", "boring"]
    return {
        "positive": {word: int(counter.get(word, 0)) for word in positive},
        "negative": {word: int(counter.get(word, 0)) for word in negative},
    }


def sample_reviews(df: pd.DataFrame, label_name: str) -> list:
    subset = df[(df["split"] == "train") & (df["label_name"] == label_name)]
    if subset.empty:
        return []
    sample = subset.sample(n=min(10, len(subset)), random_state=42)
    previews = []
    for idx, row in sample.iterrows():
        text = re.sub(r"\s+", " ", row["review"]).strip()[:500]
        previews.append(
            {
                "index": int(idx),
                "label": int(row["label"]),
                "label_name": row["label_name"],
                "file_path": row["file_path"],
                "text_preview": text,
            }
        )
    return previews


def build_report() -> dict:
    df = load_reviews()
    overview = dataset_overview(df)
    quality = text_quality(df)
    length_stats = length_statistics(df.copy())
    duplicate_stats = analyze_duplicates(df)
    vocab_info = vocabulary_analysis(df)
    neg_counts = negation_counts(vocab_info["counter"])
    sent_counts = sentiment_counts(vocab_info["counter"])
    samples = {
        "train_positive": sample_reviews(df, "pos"),
        "train_negative": sample_reviews(df, "neg"),
    }
    vocab_info.pop("counter")
    return {
        "dataset_overview": overview,
        "text_quality_checks": quality,
        "length_statistics": length_stats,
        "duplicate_analysis": duplicate_stats,
        "vocabulary_analysis": vocab_info,
        "negation_counts": neg_counts,
        "sentiment_word_counts": sent_counts,
        "sample_inspection": samples,
    }


def main() -> None:
    output_dir = ensure_dir(get_data_understanding_output_dir())
    output_path = output_dir / "imdb_data_understanding.json"
    report = build_report()
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
