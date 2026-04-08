from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "result"
DATA_DIR = RESULT_DIR / "dataProcessing"
ALG_DIR = RESULT_DIR / "algorithms"
OUTPUT_DIR = Path(__file__).resolve().parent
VISUALS_DIR = OUTPUT_DIR / "visuals"

MODEL_NAME_MAP = {
    "vader": "VADER",
    "logistic": "Logistic Regression",
    "linearsvc": "LinearSVC",
    "tree": "Decision Tree",
    "random_forest": "Random Forest",
}

LABEL_ZH = {0: "负面", 1: "正面"}

POSITIVE_LEXICON = {
    "good",
    "great",
    "amazing",
    "excellent",
    "fantastic",
    "love",
    "wonderful",
    "best",
}
NEGATIVE_LEXICON = {
    "bad",
    "awful",
    "terrible",
    "horrible",
    "hate",
    "poor",
    "worst",
    "boring",
}
NEGATION_WORDS = {
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
}

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def md_table(rows: list[dict], headers: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = []
        for h in headers:
            value = row.get(h, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            else:
                value = str(value)
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def truncate_text(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_cleaned_reviews() -> pd.DataFrame:
    return read_csv(
        DATA_DIR / "cleaned_reviews.csv",
        usecols=["review_id", "split", "label", "label_name", "raw_review", "cleaned_review"],
    )


def load_metrics() -> pd.DataFrame:
    df = read_csv(ALG_DIR / "model_metrics.csv")
    df["display_name"] = df["model"].map(MODEL_NAME_MAP).fillna(df["model"])
    return df


def load_predictions(model: str) -> pd.DataFrame:
    df = read_csv(ALG_DIR / f"{model}_test_predictions.csv")
    clean = load_cleaned_reviews()
    merged = df.merge(
        clean[["review_id", "raw_review", "cleaned_review"]],
        on="review_id",
        how="left",
        validate="one_to_one",
    )
    merged["model"] = model
    merged["display_name"] = MODEL_NAME_MAP.get(model, model)
    merged["correct"] = merged["label"] == merged["predicted_label"]
    merged["word_count_raw"] = merged["raw_review"].fillna("").str.split().apply(len)
    merged["word_count_clean"] = merged["cleaned_review"].fillna("").str.split().apply(len)
    merged["char_count_raw"] = merged["raw_review"].fillna("").str.len()
    merged["char_count_clean"] = merged["cleaned_review"].fillna("").str.len()
    merged["abs_score"] = merged["score"].abs()
    return merged


def load_all_predictions() -> pd.DataFrame:
    return pd.concat([load_predictions(model) for model in MODEL_NAME_MAP], ignore_index=True)


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def count_sentiment_tokens(tokens: list[str]) -> tuple[int, int]:
    pos = sum(1 for t in tokens if t in POSITIVE_LEXICON)
    neg = sum(1 for t in tokens if t in NEGATIVE_LEXICON)
    return pos, neg


def count_negations(tokens: list[str]) -> int:
    return sum(1 for t in tokens if t in NEGATION_WORDS)


def has_contrast_markers(text: str) -> bool:
    text = str(text).lower()
    return any(marker in text for marker in [" but ", " however ", " although ", " though ", " yet "])


def has_sarcasm_markers(text: str) -> bool:
    lowered = str(text).lower()
    return any(
        marker in lowered
        for marker in [
            "yeah right",
            "as if",
            "great job",
            "wonderful job",
            "what a joke",
        ]
    ) or "???" in text or "!!!" in text


def sentence_count(text: str) -> int:
    pieces = [p for p in SENTENCE_SPLIT_RE.split(str(text)) if p.strip()]
    return max(len(pieces), 1)


def assign_error_type(row: pd.Series) -> tuple[str, str]:
    raw = str(row.get("raw_review", ""))
    cleaned = str(row.get("cleaned_review", ""))
    tokens = tokenize(cleaned)
    pos_count, neg_count = count_sentiment_tokens(tokens)
    negations = count_negations(tokens)
    word_count = int(row.get("word_count_raw", len(raw.split())))
    abs_score = float(row.get("abs_score", 0.0))
    contrast = has_contrast_markers(raw)
    sarcasm = has_sarcasm_markers(raw)

    if word_count <= 80:
        return "Short review / sparse evidence", "Review is short, so sentiment cues are limited and the model may overreact to a few strong tokens."
    if word_count >= 500:
        return "Very long review / topic drift", "The review is long and mixes plot summary, evaluation, and side comments, which can dilute the sentiment signal."
    if abs_score <= 0.25:
        return "Borderline confidence", "The model score is close to the decision boundary, so a small lexical cue can flip the prediction."
    if negations and (pos_count > 0 or neg_count > 0):
        return "Negation scope issue", "Negation words such as not/no/never likely change the local meaning, but the bag-of-words features do not model scope well."
    if pos_count > 0 and neg_count > 0 and contrast:
        return "Mixed sentiment with contrast", "The review contains both praise and criticism joined by contrast markers such as but/however, making the dominant polarity harder to infer."
    if pos_count > 0 and neg_count > 0:
        return "Mixed sentiment", "Positive and negative sentiment words co-exist, so the review is inherently ambiguous."
    if sarcasm:
        return "Sarcasm / irony", "The review uses rhetorical emphasis or sarcastic phrasing, which surface-level lexical features usually miss."
    if sentence_count(raw) >= 12 and word_count >= 250:
        return "Long structured review", "The text looks like a long, structured critique with plot summary and evaluation mixed together."
    return "General ambiguity", "The wording is indirect or domain-specific, so the model does not see a clean sentiment cue."


def build_error_cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_predictions("linearsvc")
    errors = df[~df["correct"]].copy()
    error_rows = []
    for _, row in errors.iterrows():
        err_type, reason = assign_error_type(row)
        error_rows.append(
            {
                "review_id": int(row["review_id"]),
                "true_label": row["label_name"],
                "predicted_label": row["predicted_label_name"],
                "score": float(row["score"]),
                "word_count": int(row["word_count_raw"]),
                "char_count": int(row["char_count_raw"]),
                "error_type": err_type,
                "reason": reason,
                "review_text": str(row["raw_review"]),
            }
        )
    error_df = pd.DataFrame(error_rows)

    priority = [
        "Short review / sparse evidence",
        "Very long review / topic drift",
        "Negation scope issue",
        "Mixed sentiment with contrast",
        "Mixed sentiment",
        "Sarcasm / irony",
        "Borderline confidence",
        "Long structured review",
        "General ambiguity",
    ]
    selected: list[dict] = []
    used_ids: set[int] = set()
    for category in priority:
        subset = error_df[error_df["error_type"] == category].copy()
        subset = subset.sort_values("score", key=lambda s: s.abs(), ascending=False)
        for _, row in subset.head(2).iterrows():
            rid = int(row["review_id"])
            if rid not in used_ids:
                used_ids.add(rid)
                selected.append(row.to_dict())
            if len(selected) >= 15:
                break
        if len(selected) >= 15:
            break

    if len(selected) < 15:
        remaining = error_df[~error_df["review_id"].isin(used_ids)].copy()
        remaining = remaining.sort_values("score", key=lambda s: s.abs(), ascending=False)
        for _, row in remaining.iterrows():
            selected.append(row.to_dict())
            if len(selected) >= 15:
                break

    sample_df = pd.DataFrame(selected).head(15).copy()
    sample_df["review_excerpt"] = sample_df["review_text"].apply(lambda x: truncate_text(x, 320))
    return error_df, sample_df


def build_feature_interpretation() -> pd.DataFrame:
    rows = []
    for model in ["logistic", "linearsvc"]:
        payload = read_json(ALG_DIR / f"{model}_top_features.json")
        display = MODEL_NAME_MAP[model]
        for polarity_key, polarity in [("top_positive_features", "Positive"), ("top_negative_features", "Negative")]:
            for rank, item in enumerate(payload.get(polarity_key, []), start=1):
                rows.append(
                    {
                        "source": display,
                        "feature_type": polarity,
                        "rank": rank,
                        "feature": str(item["feature"]),
                        "weight_or_importance": float(item["weight"]),
                        "interpretation": "Strong positive cue in the TF-IDF space."
                        if polarity == "Positive"
                        else "Strong negative cue in the TF-IDF space.",
                    }
                )

    for model in ["tree", "random_forest"]:
        df = read_csv(ALG_DIR / f"{model}_feature_importances.csv")
        display = MODEL_NAME_MAP[model]
        for rank, row in enumerate(df.sort_values("importance", ascending=False).itertuples(index=False), start=1):
            rows.append(
                {
                    "source": display,
                    "feature_type": "Statistical",
                    "rank": rank,
                    "feature": row.feature,
                    "weight_or_importance": float(row.importance),
                    "interpretation": "Higher importance means the statistical feature contributes more to the tree-based split.",
                }
            )
    return pd.DataFrame(rows)


def build_distribution_analysis() -> dict:
    clean = load_cleaned_reviews()
    clean["word_count_raw"] = clean["raw_review"].fillna("").str.split().apply(len)
    clean["char_count_raw"] = clean["raw_review"].fillna("").str.len()
    clean["word_count_clean"] = clean["cleaned_review"].fillna("").str.split().apply(len)
    clean["char_count_clean"] = clean["cleaned_review"].fillna("").str.len()

    metrics = load_metrics()
    linearsvc_preds = load_predictions("linearsvc").copy()
    linearsvc_preds["correct"] = linearsvc_preds["label"] == linearsvc_preds["predicted_label"]

    bins = [0, 50, 100, 150, 200, 300, 500, 1000, 10000]
    labels = ["<=50", "51-100", "101-150", "151-200", "201-300", "301-500", "501-1000", ">1000"]
    linearsvc_preds["length_bin"] = pd.cut(linearsvc_preds["word_count_raw"], bins=bins, labels=labels, include_lowest=True)
    length_perf = (
        linearsvc_preds.groupby("length_bin", observed=False)["correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "accuracy"})
    )

    linearsvc_preds["confidence_bin"] = pd.qcut(linearsvc_preds["abs_score"], q=5, duplicates="drop")
    conf_perf = (
        linearsvc_preds.groupby("confidence_bin", observed=False)["correct"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "accuracy"})
    )

    all_preds = load_all_predictions().copy()
    all_preds["correct"] = all_preds["label"] == all_preds["predicted_label"]
    all_preds["length_bin"] = pd.cut(all_preds["word_count_raw"], bins=bins, labels=labels, include_lowest=True)
    model_length_perf = []
    for model, group in all_preds.groupby("display_name"):
        grouped = group.groupby("length_bin", observed=False)["correct"].mean().reindex(labels)
        model_length_perf.append(
            {
                "model": model,
                "series": [float(x) if pd.notna(x) else np.nan for x in grouped.tolist()],
            }
        )

    class_distribution = clean.groupby(["split", "label_name"]).size().reset_index(name="count").sort_values(["split", "label_name"])
    length_summary = {
        "raw_review": {
            "min": int(clean["char_count_raw"].min()),
            "max": int(clean["char_count_raw"].max()),
            "mean": float(clean["char_count_raw"].mean()),
            "median": float(clean["char_count_raw"].median()),
            "p95": float(clean["char_count_raw"].quantile(0.95)),
        },
        "raw_word_count": {
            "min": int(clean["word_count_raw"].min()),
            "max": int(clean["word_count_raw"].max()),
            "mean": float(clean["word_count_raw"].mean()),
            "median": float(clean["word_count_raw"].median()),
            "p95": float(clean["word_count_raw"].quantile(0.95)),
        },
        "clean_word_count": {
            "min": int(clean["word_count_clean"].min()),
            "max": int(clean["word_count_clean"].max()),
            "mean": float(clean["word_count_clean"].mean()),
            "median": float(clean["word_count_clean"].median()),
            "p95": float(clean["word_count_clean"].quantile(0.95)),
        },
    }
    return {
        "metrics": metrics,
        "class_distribution": class_distribution,
        "length_summary": length_summary,
        "length_perf": length_perf,
        "confidence_perf": conf_perf,
        "model_length_perf": model_length_perf,
        "linearsvc_preds": linearsvc_preds,
    }


def build_inventory_md() -> str:
    files = sorted(
        [
            p
            for p in PROJECT_ROOT.rglob("*")
            if p.is_file() and p.suffix.lower() != ".pyc"
        ],
        key=lambda p: str(p).lower(),
    )
    buckets: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        ext = path.suffix.lower()
        if ext == ".py":
            buckets["Python code"].append(path)
        elif ext == ".ipynb":
            buckets["Notebook"].append(path)
        elif "member_c_output" in path.parts:
            buckets["Member C outputs"].append(path)
        elif ext == ".png":
            buckets["Image artifacts"].append(path)
        elif ext in {".csv", ".json", ".md", ".txt"} and "result" in path.parts:
            buckets["Result artifacts"].append(path)
        else:
            buckets["Project files"].append(path)

    lines = ["# Project Inventory", "", "This inventory was generated by scanning the workspace tree.", "", "## File List By Type"]
    for title, group in buckets.items():
        lines.append(f"### {title}")
        for p in group:
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            purpose = ""
            if rel.endswith("preprocess_and_engineer_features.py"):
                purpose = "Member A preprocessing and feature engineering pipeline."
            elif rel.endswith("imdb_data_understanding.py"):
                purpose = "Dataset exploration and summary report generator."
            elif rel.endswith("path_utils.py"):
                purpose = "Shared path helpers."
            elif rel.endswith("model_metrics.csv"):
                purpose = "Aggregated evaluation metrics."
            elif rel.endswith("_test_predictions.csv"):
                purpose = "Per-review prediction outputs."
            elif rel.endswith("_top_features.json"):
                purpose = "Top positive and negative TF-IDF features."
            elif rel.endswith("_feature_importances.csv"):
                purpose = "Tree-based feature importance table."
            elif rel.endswith("cleaned_reviews.csv"):
                purpose = "Preprocessed review table with raw and cleaned text."
            elif rel.endswith("labels.csv"):
                purpose = "Label alignment table."
            elif rel.endswith("feature_shapes.json"):
                purpose = "Feature matrix shape summary."
            elif rel.endswith("preprocessing_log.json"):
                purpose = "Preprocessing settings and deduplication stats."
            elif rel.endswith("imdb_data_understanding.json"):
                purpose = "Exploratory data understanding report."
            elif rel.endswith("feature_readme.md"):
                purpose = "Feature engineering output summary."
            elif rel.endswith("readme.txt"):
                purpose = "Result folder notes."
            elif rel.endswith("Project2.ipynb"):
                purpose = "Project notebook scaffold."
            lines.append(f"- `{rel}`{f' - {purpose}' if purpose else ''}")
        lines.append("")

    lines.append("## Existing Completed Work")
    lines.append("- Member A: preprocessing, cleaning, TF-IDF features, and statistical features.")
    lines.append("- Member B: model training, tuning, predictions, and evaluation summaries.")
    lines.append("- Saved models, prediction CSVs, feature weights, and importances are already present.")
    lines.append("")
    lines.append("## Member C Status")
    lines.append("- The analysis/reporting deliverables are generated in `member_c_output/`.")
    lines.append("- Before this run, those deliverables were the missing part of the project.")
    lines.append("")
    lines.append("## Not Found In Workspace")
    lines.append("- No standalone `full_train.csv`, `x_baseline.csv`, or `y_baseline.csv` files were found.")
    lines.append("- Equivalent content is available in `result/dataProcessing/cleaned_reviews.csv` and `labels.csv`.")
    return "\n".join(lines)


def build_error_cases_md(error_df: pd.DataFrame, sample_df: pd.DataFrame) -> str:
    metrics = load_metrics().set_index("model")
    lines = ["# Error Cases Analysis", "", "## Overall Error Profile"]
    lines.append(
        f"- Best model used here: **LinearSVC** with Accuracy={metrics.loc['linearsvc', 'accuracy']:.4f} and F1={metrics.loc['linearsvc', 'f1']:.4f}."
    )
    lines.append(f"- Misclassified test reviews: **{len(error_df)}**")
    lines.append("")
    lines.append(md_table(
        error_df.groupby("error_type").size().reset_index(name="count").sort_values("count", ascending=False).to_dict(orient="records"),
        ["error_type", "count"],
    ))
    lines.append("## Representative Samples")
    display_rows = []
    for row in sample_df.itertuples(index=False):
        display_rows.append(
            {
                "review_id": int(row.review_id),
                "true_label": LABEL_ZH.get(1 if row.true_label == "pos" else 0),
                "predicted_label": LABEL_ZH.get(1 if row.predicted_label == "pos" else 0),
                "score": float(row.score),
                "word_count": int(row.word_count),
                "error_type": row.error_type,
                "reason": row.reason,
                "review_excerpt": row.review_excerpt,
            }
        )
    lines.append(md_table(display_rows, ["review_id", "true_label", "predicted_label", "score", "word_count", "error_type", "reason", "review_excerpt"]))
    lines.append("## Interpretation")
    lines.append("- Short reviews are fragile because a single adjective or exclamation mark can dominate the signal.")
    lines.append("- Very long reviews often mix plot summary and critique, which blurs the overall polarity.")
    lines.append("- Negation and contrast markers (`not`, `but`, `however`) are still hard for bag-of-words models.")
    lines.append("- Sarcasm and irony are especially difficult because the surface words may disagree with the real intent.")
    return "\n".join(lines)


def build_feature_interpretation_md(feature_df: pd.DataFrame) -> str:
    lines = ["# Feature Interpretation", "", "## Linear Model Keywords"]
    for source in ["LinearSVC", "Logistic Regression"]:
        sub = feature_df[(feature_df["source"] == source) & (feature_df["feature_type"].isin(["Positive", "Negative"]))]
        lines.append(f"### {source}")
        lines.append("- Positive-side features")
        lines.append(md_table(sub[sub["feature_type"] == "Positive"].head(12).to_dict(orient="records"), ["rank", "feature", "weight_or_importance", "interpretation"]))
        lines.append("- Negative-side features")
        lines.append(md_table(sub[sub["feature_type"] == "Negative"].head(12).to_dict(orient="records"), ["rank", "feature", "weight_or_importance", "interpretation"]))
    lines.append("## Tree-Based Statistical Features")
    for source in ["Decision Tree", "Random Forest"]:
        sub = feature_df[(feature_df["source"] == source) & (feature_df["feature_type"] == "Statistical")].head(8)
        lines.append(f"### {source}")
        lines.append(md_table(sub.to_dict(orient="records"), ["rank", "feature", "weight_or_importance", "interpretation"]))
    lines.append("## Takeaways")
    lines.append("- Positive cues include `great`, `excellent`, `perfect`, `love`, and `fantastic`.")
    lines.append("- Negative cues include `bad`, `worst`, `awful`, `boring`, and `terrible`.")
    lines.append("- Tree models rely heavily on `negative_word_count` and `positive_word_count`, confirming the value of simple sentiment counts.")
    lines.append("- Stemming compresses variants into tokens such as `excel`, `amaz`, and `terribl`, which helps linear models generalize.")
    return "\n".join(lines)


def build_distribution_md(dist: dict) -> str:
    metrics = dist["metrics"]
    lines = ["# Distribution Analysis", "", "## Dataset Distribution"]
    lines.append(md_table(dist["class_distribution"].to_dict(orient="records"), ["split", "label_name", "count"]))
    ls = dist["length_summary"]
    lines.append("## Review Length Summary")
    lines.append(f"- Raw review character count: mean={ls['raw_review']['mean']:.1f}, median={ls['raw_review']['median']:.1f}, p95={ls['raw_review']['p95']:.1f}")
    lines.append(f"- Raw word count: mean={ls['raw_word_count']['mean']:.1f}, median={ls['raw_word_count']['median']:.1f}, p95={ls['raw_word_count']['p95']:.1f}")
    lines.append(f"- Cleaned word count: mean={ls['clean_word_count']['mean']:.1f}, median={ls['clean_word_count']['median']:.1f}, p95={ls['clean_word_count']['p95']:.1f}")
    lines.append("")
    lp = dist["length_perf"].copy()
    lp["error_rate"] = 1 - lp["accuracy"]
    lines.append("## LinearSVC Accuracy By Length Bin")
    lines.append(md_table(
        lp.assign(accuracy=lp["accuracy"].map(lambda x: f"{x:.4f}"), error_rate=lp["error_rate"].map(lambda x: f"{x:.4f}")).to_dict(orient="records"),
        ["length_bin", "count", "accuracy", "error_rate"],
    ))
    cp = dist["confidence_perf"].copy()
    cp["error_rate"] = 1 - cp["accuracy"]
    lines.append("## Confidence vs Error")
    lines.append("For LinearSVC, the absolute decision margin is a good confidence proxy: small margins are much more error-prone.")
    lines.append(md_table(
        cp.assign(accuracy=cp["accuracy"].map(lambda x: f"{x:.4f}"), error_rate=cp["error_rate"].map(lambda x: f"{x:.4f}")).to_dict(orient="records"),
        ["confidence_bin", "count", "accuracy", "error_rate"],
    ))
    lines.append("### Key Trend")
    lines.append("The lowest-confidence quintile is only about two-thirds correct, while the highest-confidence quintile is almost always correct.")
    lines.append("")
    lines.append("## Model Snapshot")
    lines.append(md_table(metrics.to_dict(orient="records"), ["display_name", "accuracy", "precision", "recall", "f1"]))
    return "\n".join(lines)


def build_final_report_md(dist: dict, feature_df: pd.DataFrame) -> str:
    metrics = dist["metrics"].copy()
    metrics["display_name"] = metrics["model"].map(MODEL_NAME_MAP)
    best_row = metrics.sort_values("f1", ascending=False).iloc[0]
    best_model = best_row["display_name"]
    best_f1 = float(best_row["f1"])
    best_acc = float(best_row["accuracy"])
    top_words = feature_df[(feature_df["source"] == "LinearSVC") & (feature_df["feature_type"] == "Positive")].head(5)["feature"].tolist()
    neg_words = feature_df[(feature_df["source"] == "LinearSVC") & (feature_df["feature_type"] == "Negative")].head(5)["feature"].tolist()
    lines = ["# Final Report Draft", ""]
    lines += [
        "## 1. 项目背景与研究目标",
        "本项目围绕电影评论情感分析展开，目标是从文本中判断评论正负情绪，并比较不同模型在相同测试集上的表现。",
        "",
        "## 2. 数据集说明",
        "根据现有数据理解结果，原始数据集是平衡的二分类评论集合，正负样本各占一半；成员 A 的预处理保留了 raw text 和 cleaned text 以便后续分析。",
        "当前 workspace 未找到单独的 `full_train.csv`、`x_baseline.csv`、`y_baseline.csv`，但 `cleaned_reviews.csv` 和 `labels.csv` 已保存对应信息。",
        "",
        "## 3. Data Mining 整体流程",
        "原始文本 -> 清洗标准化 -> 特征工程 -> 模型训练与调参 -> 测试集评估 -> 错误分析 -> 特征解释 -> 汇报呈现。",
        "",
        "## 4. 数据预处理与特征工程（成员 A）",
        "成员 A 完成了 HTML 清理、大小写统一、数字处理、词干化、去重，以及 unigram / bigram TF-IDF 与统计特征构建。",
        "其中 `features_v3_statistical.csv` 包含字符数、词数、平均词长、感叹号数、问号数、否定词计数和正负向词计数。",
        "",
        "## 5. 模型构建与优化（成员 B）",
        "成员 B 训练并比较了 VADER、Logistic Regression、LinearSVC、Decision Tree 和 Random Forest，并保留了调参结果、预测结果和特征解释文件。",
        f"从测试集表现看，`{best_model}` 是当前最强模型，Accuracy={best_acc:.4f}，F1={best_f1:.4f}；Logistic Regression 紧随其后，F1 也达到 0.8938。",
        "VADER 作为词典方法召回率较高，但精度明显偏低；树模型和随机森林受限于统计特征表达能力，整体弱于线性文本模型。",
        "",
        "## 6. 模型结果分析",
        "线性模型显著优于基于词典和浅层统计特征的模型，说明该任务更依赖高维稀疏词项和短语信息。",
        "LinearSVC 与 Logistic Regression 的表现非常接近，说明在当前特征下任务已经接近一个稳定的线性分界。",
        "LinearSVC 的错误主要集中在低置信度样本和长评论样本上：当决策边界附近的 margin 较小时，错误率明显升高。",
        "",
        "## 7. 错误案例分析",
        "主要误差来源包括：短评论信息不足、长评论主题漂移、否定词作用范围、正负混合情绪以及反讽/语气反转。",
        "这说明仅靠词袋特征和线性边界无法完整捕捉语义组合关系，尤其是 `not good`、`but` 转折句和带讽刺意味的表述。",
        "",
        "## 8. 特征重要性解读",
        "正向词包括 " + "、".join(top_words) + "；负向词包括 " + "、".join(neg_words) + "。",
        "树模型的统计特征重要性也表明 `negative_word_count` 和 `positive_word_count` 是最关键的两项。",
        "",
        "## 9. 可视化结果说明",
        "本次交付已生成流程图、模型对比图、错误类型分布图、评论长度分布图、不同长度下的模型表现图，以及正负向关键词图。",
        "",
        "## 10. 关键发现",
        "- 线性文本模型优于词典法和树模型。",
        "- 关键词和短语信息比单纯统计特征更有判别力。",
        "- 长评论和低 margin 样本更容易出错。",
        "- 否定、转折和反讽是主要误差来源。",
        "",
        "## 11. 存在问题与改进建议",
        "- 可尝试上下文模型、embedding 或 Transformer。",
        "- 可针对否定范围和转折结构设计显式特征。",
        "- 可使用校准方法或阈值调整改进边界样本稳定性。",
        "",
        "## 12. 小组分工说明",
        "- 成员 A：数据预处理与特征工程。",
        "- 成员 B：模型构建与优化。",
        "- 成员 C：结果分析、错误案例、特征解读、可视化、报告整合与汇报材料。",
    ]
    return "\n".join(lines)


def build_presentation_outline_md() -> str:
    slides = [
        ("1. 标题页：电影评论情感分析", ["项目名称、课程名、小组成员分工", "说明任务是对电影评论做正负情感分类", "展示总览图或流程图"], "流程图", "先用一句话说明任务，再概括文本分类路线。"),
        ("2. 研究背景与目标", ["情感分析是典型文本挖掘任务", "目标是比较多个模型的分类效果", "重点关注 Accuracy、F1 和可解释性"], "数据集说明图", "强调这是一个平衡二分类任务。"),
        ("3. 数据集与预处理", ["数据来自电影评论，正负样本均衡", "完成 HTML 清理、大小写统一、词干化和去重", "保留否定词和情绪词"], "review_length_distribution.png", "先讲数据，再解释预处理如何降噪。"),
        ("4. 特征工程", ["TF-IDF unigram / bigram 捕捉词项和短语", "统计特征补充字符数、词数、否定词和情绪词计数", "线性模型和树模型使用不同特征"], "关键词图", "说明为什么文本任务既需要高维稀疏特征，也需要少量统计特征。"),
        ("5. 模型候选与调参", ["比较 VADER、Logistic Regression、LinearSVC、Decision Tree、Random Forest", "线性模型做了参数搜索", "记录交叉验证结果"], "model_performance_comparison.png", "把模型比较作为核心过渡页。"),
        ("6. 最佳模型结果", ["当前最优模型为 LinearSVC", "F1 接近 0.90", "显著高于词典法和树模型"], "model_performance_comparison.png", "直接读出最关键数字。"),
        ("7. 不同长度下的表现", ["短评论通常更容易分类", "长评论更容易出现主题漂移和混合情绪", "边界样本的置信度更低"], "performance_by_length.png", "把长度和错误联系起来。"),
        ("8. 错误案例分析", ["短评论信息不足", "否定词和转折结构导致误判", "反讽、双关和混合情绪是主要难点"], "error_type_distribution.png", "选 2-3 条典型错分评论说明原因。"),
        ("9. 特征重要性与关键词", ["正向词：great、excellent、perfect、love", "负向词：bad、worst、awful、boring", "树模型强调 negative_word_count 和 positive_word_count"], "positive_keywords.png / negative_keywords.png", "告诉听众哪些词最有影响力。"),
        ("10. 关键发现与改进方向", ["线性模型优于词典法与树模型", "否定、转折和讽刺仍是主要错误来源", "可尝试上下文模型、否定范围建模和阈值校准"], "总结页", "用三句话收尾：做得最好、哪里错、下一步怎么改。"),
        ("11. 小组分工与交付", ["成员 A：预处理与特征工程", "成员 B：模型构建与优化", "成员 C：分析、可视化、报告和展示材料"], "团队分工表", "最后说明每位成员负责的部分。"),
    ]
    lines = ["# Presentation Outline", ""]
    for title, points, figure, notes in slides:
        lines.append(f"## {title}")
        lines.append("### Key Points")
        lines.extend([f"- {point}" for point in points])
        lines.append("### Suggested Visual")
        lines.append(f"- {figure}")
        lines.append("### Speaker Notes")
        lines.append(f"- {notes}")
        lines.append("")
    return "\n".join(lines)


def build_readme_md() -> str:
    lines = ["# README for Member C Outputs", "", "## Input Files Read"]
    lines += [
        "- `result/dataProcessing/cleaned_reviews.csv`",
        "- `result/dataProcessing/labels.csv`",
        "- `result/dataProcessing/features_v3_statistical.csv`",
        "- `result/dataProcessing/preprocessing_log.json`",
        "- `result/dataProcessing/feature_shapes.json`",
        "- `result/dataProcessing/feature_readme.md`",
        "- `result/dataUnderstanding/imdb_data_understanding.json`",
        "- `result/dataUnderstanding/imdb_data_understanding.md`",
        "- `result/algorithms/model_metrics.csv` and `model_metrics.json`",
        "- `result/algorithms/*_test_predictions.csv`",
        "- `result/algorithms/*_top_features.json`",
        "- `result/algorithms/*_feature_importances.csv`",
        "- `result/algorithms/*_cv_results.csv`",
    ]
    lines += ["", "## New Files Generated", "- `project_inventory.md`", "- `error_cases_analysis.csv` / `error_cases_analysis.md`", "- `feature_interpretation.csv` / `feature_interpretation.md`", "- `distribution_analysis.md`", "- `final_report_draft.md`", "- `presentation_outline.md`", "- `generate_visuals.py`", "- `analyze_results.py`", "- `visuals/` PNG charts", "", "## Purpose of Each File", "- `project_inventory.md`: file scan and project status summary.", "- `error_cases_analysis.*`: representative misclassified examples and error type breakdown.", "- `feature_interpretation.*`: word-level and statistical feature explanations.", "- `distribution_analysis.md`: length distribution, class balance, and confidence/error analysis.", "- `final_report_draft.md`: submission-ready report draft in Chinese.", "- `presentation_outline.md`: 8-12 slide talk outline for a 5-8 minute presentation.", "- `generate_visuals.py`: reproducible plotting script.", "- `analyze_results.py`: reproducible analysis and report-generation script.", "", "## Reproduction Order", "1. Run `python analyze_results.py` from the `member_c_output/` directory.", "2. Run `python generate_visuals.py` from the same directory.", "3. Insert the PNG files from `visuals/` into the final report or PPT."]
    return "\n".join(lines)


def write_outputs():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(VISUALS_DIR)
    error_df, sample_df = build_error_cases()
    error_df.to_csv(OUTPUT_DIR / "error_cases_analysis.csv", index=False, encoding="utf-8")
    write_text(OUTPUT_DIR / "error_cases_analysis.md", build_error_cases_md(error_df, sample_df))
    feature_df = build_feature_interpretation()
    feature_df.to_csv(OUTPUT_DIR / "feature_interpretation.csv", index=False, encoding="utf-8")
    write_text(OUTPUT_DIR / "feature_interpretation.md", build_feature_interpretation_md(feature_df))
    dist = build_distribution_analysis()
    write_text(OUTPUT_DIR / "distribution_analysis.md", build_distribution_md(dist))
    write_text(OUTPUT_DIR / "final_report_draft.md", build_final_report_md(dist, feature_df))
    write_text(OUTPUT_DIR / "presentation_outline.md", build_presentation_outline_md())
    write_text(OUTPUT_DIR / "README_member_c.md", build_readme_md())
    write_text(OUTPUT_DIR / "project_inventory.md", build_inventory_md())
    return {"error_rows": len(error_df), "error_samples": len(sample_df), "feature_rows": len(feature_df)}


def main() -> None:
    print(json.dumps(write_outputs(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
