# Feature Engineering Deliverables

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
