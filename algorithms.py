import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
except ImportError:  # pragma: no cover - handled at runtime
    SentimentIntensityAnalyzer = None
    nltk = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "result" / "dataProcessing"
OUTPUT_DIR = PROJECT_ROOT / "result" / "algorithms"

LABEL_COLS = ["review_id", "split", "label", "label_name"]
STAT_COLS = [
    "char_count",
    "word_count",
    "avg_word_length",
    "exclaim_count",
    "question_count",
    "negation_count",
    "positive_word_count",
    "negative_word_count",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sparse_if_exists(path: Path):
    if path.exists():
        return sparse.load_npz(path)
    return None


def load_inputs():
    cleaned_reviews = pd.read_csv(DATA_DIR / "cleaned_reviews.csv")
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    v3_df = pd.read_csv(DATA_DIR / "features_v3_statistical.csv")

    if len(cleaned_reviews) != len(labels) or len(v3_df) != len(labels):
        raise ValueError("Input files are not aligned. Check rows in cleaned_reviews, labels, and V3.")

    train_mask = labels["split"].eq("train").to_numpy()
    test_mask = labels["split"].eq("test").to_numpy()
    y_train = labels.loc[train_mask, "label"].to_numpy()
    y_test = labels.loc[test_mask, "label"].to_numpy()

    x_v2_train = load_sparse_if_exists(DATA_DIR / "features_v2_train_tfidf_uni_bigram.npz")
    x_v2_test = load_sparse_if_exists(DATA_DIR / "features_v2_test_tfidf_uni_bigram.npz")

    if x_v2_train is not None and x_v2_test is not None:
        if x_v2_train.shape[0] != train_mask.sum() or x_v2_test.shape[0] != test_mask.sum():
            raise ValueError("Split V2 files are not aligned with labels.csv.")
    else:
        x_v2_combined = load_sparse_if_exists(DATA_DIR / "features_v2_tfidf_uni_bigram.npz")
        if x_v2_combined is None:
            raise FileNotFoundError("Cannot find V2 TF-IDF files in result/dataProcessing.")
        if x_v2_combined.shape[0] != len(labels):
            raise ValueError("Combined V2 file is not aligned with labels.csv.")
        x_v2_train = x_v2_combined[train_mask]
        x_v2_test = x_v2_combined[test_mask]

    return cleaned_reviews, labels, x_v2_train, x_v2_test, v3_df, train_mask, test_mask, y_train, y_test


def get_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_predictions(
    model_name: str,
    labels: pd.DataFrame,
    test_mask: np.ndarray,
    preds,
    scores=None,
):
    output = labels.loc[test_mask, LABEL_COLS].copy()
    output["predicted_label"] = preds
    output["predicted_label_name"] = np.where(output["predicted_label"] == 1, "pos", "neg")
    if scores is not None:
        output["score"] = scores
    output.to_csv(OUTPUT_DIR / f"{model_name}_test_predictions.csv", index=False)


def save_metrics(results):
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def build_cv_summary(search):
    cv_df = pd.DataFrame(search.cv_results_)
    keep_cols = ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    summary = cv_df[keep_cols].sort_values("rank_test_score").copy()
    summary["params"] = summary["params"].apply(lambda x: json.dumps(x, sort_keys=True))
    return summary


def run_vader(reviews, labels, test_mask, y_test):
    model_name = "vader"

    if SentimentIntensityAnalyzer is None or nltk is None:
        result = {
            "model": model_name,
            "status": "failed",
            "reason": "nltk is not installed in this environment.",
        }
        placeholder = labels.loc[test_mask, LABEL_COLS].copy()
        placeholder["predicted_label"] = np.nan
        placeholder["predicted_label_name"] = ""
        placeholder["score"] = np.nan
        placeholder["error_reason"] = result["reason"]
        placeholder.to_csv(OUTPUT_DIR / f"{model_name}_test_predictions.csv", index=False)
        (OUTPUT_DIR / "vader_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    try:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        result = {
            "model": model_name,
            "status": "failed",
            "reason": "VADER lexicon is missing. Install/download nltk vader_lexicon before running.",
        }
        placeholder = labels.loc[test_mask, LABEL_COLS].copy()
        placeholder["predicted_label"] = np.nan
        placeholder["predicted_label_name"] = ""
        placeholder["score"] = np.nan
        placeholder["error_reason"] = result["reason"]
        placeholder.to_csv(OUTPUT_DIR / f"{model_name}_test_predictions.csv", index=False)
        (OUTPUT_DIR / "vader_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    vader = SentimentIntensityAnalyzer()
    test_reviews = reviews.loc[test_mask, "raw_review"].fillna("").tolist()
    scores = [vader.polarity_scores(text)["compound"] for text in test_reviews]
    y_pred = np.where(np.array(scores) >= 0.05, 1, 0)

    metrics = get_metrics(y_test, y_pred)
    save_predictions(model_name, labels, test_mask, y_pred, scores)

    result = {"model": model_name, "status": "success", **metrics}
    (OUTPUT_DIR / "vader_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_logistic(labels, x_train, x_test, test_mask, y_train, y_test):
    model_name = "logistic"

    search = GridSearchCV(
        estimator=LogisticRegression(max_iter=2000, random_state=42, n_jobs=1),
        param_grid={
            "C": [0.5, 1.0, 2.0, 4.0],
            "solver": ["liblinear", "lbfgs"],
        },
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]

    metrics = get_metrics(y_test, y_pred)
    save_predictions(model_name, labels, test_mask, y_pred, scores)
    joblib.dump(model, OUTPUT_DIR / "logistic.joblib")
    build_cv_summary(search).to_csv(OUTPUT_DIR / "logistic_cv_results.csv", index=False)

    top_feature_summary = {}
    feature_path = DATA_DIR / "features_v2_feature_names.json"
    if feature_path.exists():
        feature_names = np.array(json.loads(feature_path.read_text(encoding="utf-8")))
        weights = model.coef_[0]
        top_pos = np.argsort(weights)[-20:][::-1]
        top_neg = np.argsort(weights)[:20]
        top_feature_summary = {
            "top_positive_features": [
                {"feature": str(feature_names[i]), "weight": float(weights[i])} for i in top_pos
            ],
            "top_negative_features": [
                {"feature": str(feature_names[i]), "weight": float(weights[i])} for i in top_neg
            ],
        }
        (OUTPUT_DIR / "logistic_top_features.json").write_text(
            json.dumps(top_feature_summary, indent=2),
            encoding="utf-8",
        )

    result = {
        "model": model_name,
        "status": "success",
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        **metrics,
    }
    (OUTPUT_DIR / "logistic_summary.json").write_text(
        json.dumps({**result, **top_feature_summary}, indent=2),
        encoding="utf-8",
    )
    return result


def run_linearsvc(labels, x_train, x_test, test_mask, y_train, y_test):
    model_name = "linearsvc"

    search = GridSearchCV(
        estimator=LinearSVC(random_state=42),
        param_grid={"C": [0.25, 0.5, 1.0, 2.0, 4.0]},
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(x_test)
    scores = model.decision_function(x_test)

    metrics = get_metrics(y_test, y_pred)
    save_predictions(model_name, labels, test_mask, y_pred, scores)
    joblib.dump(model, OUTPUT_DIR / "linearsvc.joblib")
    build_cv_summary(search).to_csv(OUTPUT_DIR / "linearsvc_cv_results.csv", index=False)

    top_feature_summary = {}
    feature_path = DATA_DIR / "features_v2_feature_names.json"
    if feature_path.exists():
        feature_names = np.array(json.loads(feature_path.read_text(encoding="utf-8")))
        weights = model.coef_[0]
        top_pos = np.argsort(weights)[-20:][::-1]
        top_neg = np.argsort(weights)[:20]
        top_feature_summary = {
            "top_positive_features": [
                {"feature": str(feature_names[i]), "weight": float(weights[i])} for i in top_pos
            ],
            "top_negative_features": [
                {"feature": str(feature_names[i]), "weight": float(weights[i])} for i in top_neg
            ],
        }
        (OUTPUT_DIR / "linearsvc_top_features.json").write_text(
            json.dumps(top_feature_summary, indent=2),
            encoding="utf-8",
        )

    result = {
        "model": model_name,
        "status": "success",
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        **metrics,
    }
    (OUTPUT_DIR / "linearsvc_summary.json").write_text(
        json.dumps({**result, **top_feature_summary}, indent=2),
        encoding="utf-8",
    )
    return result


def run_tree(labels, stat_df, train_mask, test_mask, y_train, y_test):
    model_name = "tree"
    x_all = stat_df[STAT_COLS].to_numpy()
    x_train = x_all[train_mask]
    x_test = x_all[test_mask]

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]

    metrics = get_metrics(y_test, y_pred)
    save_predictions(model_name, labels, test_mask, y_pred, scores)
    joblib.dump(model, OUTPUT_DIR / "tree.joblib")

    feature_importances = pd.DataFrame(
        {
            "feature": STAT_COLS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importances.to_csv(OUTPUT_DIR / "tree_feature_importances.csv", index=False)

    result = {"model": model_name, "status": "success", **metrics}
    (OUTPUT_DIR / "tree_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def run_random_forest(labels, stat_df, train_mask, test_mask, y_train, y_test):
    model_name = "random_forest"
    x_all = stat_df[STAT_COLS].to_numpy()
    x_train = x_all[train_mask]
    x_test = x_all[test_mask]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]

    metrics = get_metrics(y_test, y_pred)
    save_predictions(model_name, labels, test_mask, y_pred, scores)
    joblib.dump(model, OUTPUT_DIR / "random_forest.joblib")

    feature_importances = pd.DataFrame(
        {
            "feature": STAT_COLS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importances.to_csv(OUTPUT_DIR / "random_forest_feature_importances.csv", index=False)

    result = {"model": model_name, "status": "success", **metrics}
    (OUTPUT_DIR / "random_forest_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def write_run_manifest(results):
    manifest = {
        "inputs": {
            "cleaned_reviews": str(DATA_DIR / "cleaned_reviews.csv"),
            "labels": str(DATA_DIR / "labels.csv"),
            "features_v2": str(DATA_DIR / "features_v2_tfidf_uni_bigram.npz"),
            "features_v3": str(DATA_DIR / "features_v3_statistical.csv"),
        },
        "outputs": {
            "metrics_csv": str(OUTPUT_DIR / "model_metrics.csv"),
            "metrics_json": str(OUTPUT_DIR / "model_metrics.json"),
            "prediction_files": [
                str(OUTPUT_DIR / "vader_test_predictions.csv"),
                str(OUTPUT_DIR / "logistic_test_predictions.csv"),
                str(OUTPUT_DIR / "linearsvc_test_predictions.csv"),
                str(OUTPUT_DIR / "tree_test_predictions.csv"),
                str(OUTPUT_DIR / "random_forest_test_predictions.csv"),
            ],
        },
        "notes": [
            "Predictions are for the test split only.",
            "VADER uses raw_review text from cleaned_reviews.csv.",
            "Logistic regression and LinearSVC use teammate uni+bigram TF-IDF features.",
            "Decision tree and random forest use teammate statistical features.",
        ],
        "results": results,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    ensure_output_dir()
    reviews, labels, x_train, x_test, stat_df, train_mask, test_mask, y_train, y_test = load_inputs()

    results = [
        run_vader(reviews, labels, test_mask, y_test),
        run_logistic(labels, x_train, x_test, test_mask, y_train, y_test),
        run_linearsvc(labels, x_train, x_test, test_mask, y_train, y_test),
        run_tree(labels, stat_df, train_mask, test_mask, y_train, y_test),
        run_random_forest(labels, stat_df, train_mask, test_mask, y_train, y_test),
    ]

    save_metrics(results)
    write_run_manifest(results)

    print(f"Saved algorithm outputs to: {OUTPUT_DIR}")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
