# Predictive Modeling Results

Run `python src/model.py` to generate the evaluation metrics and plots for the current dataset.

## Evaluation outputs

The pipeline produces:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices for both classifiers
- ROC curve comparison
- Random Forest feature-importance plot

## Model selection

The final model is selected using ROC-AUC on the held-out test set. Because the included dataset is intentionally small and educational, the reported metrics should be treated as demonstration results rather than production estimates.

## Generated files

After execution, the following images are created under `reports/plots/`:

- `decision_tree_confusion_matrix.png`
- `random_forest_confusion_matrix.png`
- `roc_curve_comparison.png`
- `random_forest_feature_importance.png`

## Interpretation

This project demonstrates the complete supervised-learning lifecycle: data validation, feature/target separation, stratified train/test splitting, preprocessing, model training, probability-based evaluation, visualization, and model selection.
