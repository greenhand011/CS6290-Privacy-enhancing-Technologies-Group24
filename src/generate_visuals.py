from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from analyze_results import (
    ALG_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    VISUALS_DIR,
    MODEL_NAME_MAP,
    build_distribution_analysis,
    build_error_cases,
    build_feature_interpretation,
    ensure_dir,
    load_cleaned_reviews,
    load_metrics,
)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    ensure_dir(VISUALS_DIR)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / name, bbox_inches="tight")
    plt.close(fig)


def model_display_order(metrics: pd.DataFrame) -> list[str]:
    return metrics.sort_values("f1", ascending=False)["display_name"].tolist()


def draw_pipeline_flow() -> None:
    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.axis("off")
    boxes = [
        ("Raw Data", "Original review text\nand sentiment labels"),
        ("Preprocessing", "HTML removal, lowercasing,\nstemming, deduplication"),
        ("Feature Engineering", "TF-IDF unigram/bigram\nand statistical features"),
        ("Model Training", "Logistic Regression,\nLinearSVC, Tree, RF, VADER"),
        ("Evaluation", "Accuracy, Precision,\nRecall, F1"),
        ("Error Analysis", "Misclassified cases,\nconfidence, error types"),
        ("Final Conclusion", "Best model + key findings\nand improvement ideas"),
    ]
    xs = np.linspace(0.06, 0.94, len(boxes))
    y = 0.5
    width = 0.12
    height = 0.42
    for i, (title, body) in enumerate(boxes):
        x = xs[i] - width / 2
        rect = FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5,
            edgecolor="#2a5caa",
            facecolor="#eef4ff",
        )
        ax.add_patch(rect)
        ax.text(xs[i], y + 0.09, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#17324d")
        ax.text(xs[i], y - 0.05, body, ha="center", va="center", fontsize=8.6, color="#2f3b4c")
        if i < len(boxes) - 1:
            arrow = FancyArrowPatch(
                (x + width, y),
                (xs[i + 1] - width / 2, y),
                arrowstyle="->",
                mutation_scale=16,
                linewidth=1.5,
                color="#4b5563",
            )
            ax.add_patch(arrow)
    ax.set_title("Data Mining Pipeline", pad=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "dm_pipeline_flow.png")


def draw_model_performance(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6))
    display_order = model_display_order(metrics)
    ordered = metrics.set_index("display_name").loc[display_order].reset_index()
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(ordered))
    width = 0.18
    palette = ["#356ae6", "#58b368", "#f39c12", "#c0392b"]
    for idx, col in enumerate(metric_cols):
        bars = ax.bar(x + (idx - 1.5) * width, ordered[col], width=width, label=col.capitalize(), color=palette[idx])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["display_name"], rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(ncol=4, frameon=True)
    save(fig, "model_performance_comparison.png")


def draw_error_distribution(error_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = error_df.groupby("error_type").size().sort_values(ascending=True)
    bars = ax.barh(counts.index, counts.values, color="#c74d4d")
    ax.set_xlabel("Count")
    ax.set_title("Error Type Distribution (LinearSVC)")
    ax.bar_label(bars, padding=3, fontsize=9)
    save(fig, "error_type_distribution.png")


def draw_length_distribution(cleaned: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    cleaned = cleaned.copy()
    cleaned["word_count_raw"] = cleaned["raw_review"].fillna("").str.split().apply(len)
    bins = np.linspace(0, max(1500, cleaned["word_count_raw"].max()), 50)
    ax.hist(
        cleaned.loc[cleaned["split"] == "train", "word_count_raw"],
        bins=bins,
        alpha=0.6,
        label="Train",
        color="#2a6fdb",
    )
    ax.hist(
        cleaned.loc[cleaned["split"] == "test", "word_count_raw"],
        bins=bins,
        alpha=0.55,
        label="Test",
        color="#e07a2f",
    )
    ax.set_xlabel("Review length (word count)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Review Length Distribution")
    ax.legend()
    save(fig, "review_length_distribution.png")


def draw_performance_by_length(dist: dict) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6))
    bins = dist["length_perf"]["length_bin"].tolist()
    x = np.arange(len(bins))
    palette = ["#356ae6", "#58b368", "#f39c12", "#c0392b", "#6f42c1"]
    for idx, item in enumerate(dist["model_length_perf"]):
        series = np.array(item["series"], dtype=float)
        ax.plot(x, series, marker="o", linewidth=2.2, label=item["model"], color=palette[idx % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylim(0.55, 1.0)
    ax.set_xlabel("Review length bin (word count)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance by Review Length")
    ax.legend(ncol=2)
    save(fig, "performance_by_length.png")


def draw_keyword_chart(feature_df: pd.DataFrame, polarity: str, filename: str, title: str) -> None:
    sub = feature_df[(feature_df["source"] == "LinearSVC") & (feature_df["feature_type"] == polarity)].copy()
    sub = sub.head(15).sort_values("weight_or_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    color = "#2a6fdb" if polarity == "Positive" else "#c74d4d"
    bars = ax.barh(sub["feature"], sub["weight_or_importance"], color=color)
    ax.set_title(title)
    ax.set_xlabel("LinearSVC coefficient")
    ax.bar_label(bars, padding=3, fontsize=8, fmt="%.2f")
    save(fig, filename)


def main() -> None:
    metrics = load_metrics()
    cleaned = load_cleaned_reviews()
    error_df, _ = build_error_cases()
    dist = build_distribution_analysis()
    feature_df = build_feature_interpretation()

    draw_pipeline_flow()
    draw_model_performance(metrics)
    draw_error_distribution(error_df)
    draw_length_distribution(cleaned)
    draw_performance_by_length(dist)
    draw_keyword_chart(feature_df, "Positive", "positive_keywords.png", "Top Positive Keywords")
    draw_keyword_chart(feature_df, "Negative", "negative_keywords.png", "Top Negative Keywords")

    print(f"Saved charts to {VISUALS_DIR}")


if __name__ == "__main__":
    main()
