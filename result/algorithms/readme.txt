Result folder for visualization:
C:\Users\GODzh\Desktop\CS5483\result\algorithms

Main files:
- model_metrics.csv
  Summary table for all models.
  Use for bar charts comparing accuracy, precision, recall, and f1.

- model_metrics.json
  Same metrics in JSON format.
  Use if loading by code is easier than CSV.

Prediction files:
- vader_test_predictions.csv
  Test-set predictions from VADER.
  Use for confusion matrix, prediction count chart, score histogram, and correct vs wrong chart.

- logistic_test_predictions.csv
  Test-set predictions from Logistic Regression.
  Use for confusion matrix, score histogram, positive vs negative prediction counts, and error analysis.

- linearsvc_test_predictions.csv
  Test-set predictions from LinearSVC.
  The score column is a decision value, not probability.
  Use for confusion matrix, prediction comparison, and score distribution.

- tree_test_predictions.csv
  Test-set predictions from Decision Tree.
  Use for confusion matrix, correct vs wrong chart, and score distribution.

- random_forest_test_predictions.csv
  Test-set predictions from Random Forest.
  Use for confusion matrix, correct vs wrong chart, and score distribution.

Model summary files:
- vader_summary.json
- logistic_summary.json
- linearsvc_summary.json
- tree_summary.json
- random_forest_summary.json
  These store each model's summary results.
  Use if a separate chart or report section is needed for one model.

Feature explanation files:
- logistic_top_features.json
  Top positive and negative words/phrases for logistic.
  Best for horizontal bar charts of important words.

- linearsvc_top_features.json
  Top positive and negative words/phrases for LinearSVC.
  Best for horizontal bar charts of important words.

- tree_feature_importances.csv
  Importance of each statistical feature in the decision tree.
  Best for horizontal bar chart.

- random_forest_feature_importances.csv
  Importance of each statistical feature in the random forest.
  Best for horizontal bar chart.

Cross-validation files:
- logistic_cv_results.csv
  Cross-validation tuning results for logistic.
  Use if you want a chart comparing parameter settings.

- linearsvc_cv_results.csv
  Cross-validation tuning results for LinearSVC.
  Use if you want a chart comparing parameter settings.

Run description:
- run_manifest.json
  Lists input files, output files, and notes about the run.
  Use as reference to understand what each file means.

Saved model files:
- logistic.joblib
- linearsvc.joblib
- tree.joblib
- random_forest.joblib
  These are saved models.
  Use only if you want to inspect the model in code or draw the decision tree.

Best graphs to create:
1. Model comparison bar chart
   Use model_metrics.csv
   Show accuracy, precision, recall, and f1 for all models.

2. Confusion matrix heatmaps
   Use each *_test_predictions.csv file
   Compare true label and predicted label.

3. Top positive and negative word chart
   Use logistic_top_features.json
   Show highest positive-weight and negative-weight words/phrases.

4. Feature importance chart
   Use tree_feature_importances.csv or random_forest_feature_importances.csv
   Show which statistical features matter most.

5. Decision tree graph
   Use tree.joblib
   Better to visualize only the top few levels because the full tree may be too crowded.
