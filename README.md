# Predictive Modeling Using Machine Learning

An end-to-end supervised machine learning project that predicts whether a customer is likely to purchase a product based on demographic and engagement features.

## Objective

Build and compare classification models, evaluate their performance on unseen data, and explain which model is most suitable for prediction.

## Key Features

- Data preprocessing and validation
- Train/test split
- Feature scaling where appropriate
- Decision Tree classifier
- Random Forest classifier
- Model comparison using accuracy, precision, recall, F1-score, and ROC-AUC
- Confusion matrix visualization
- ROC curve visualization
- Random Forest feature-importance analysis
- Reproducible Python workflow

## Dataset

The project includes a small, self-contained customer marketing dataset with features such as age, annual income, years of experience, website visits, average session duration, and whether the customer purchased the product.

The target variable is `purchased`:

- `0` = did not purchase
- `1` = purchased

## Project Structure

```text
Predictive-Modeling-Machine-Learning/
├── data/
│   └── customer_purchase_data.csv
├── notebooks/
│   └── predictive_modeling.ipynb
├── src/
│   └── model.py
├── reports/
│   └── model_results.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Machine Learning Workflow

1. Load the dataset.
2. Inspect data types, missing values, duplicates, and class distribution.
3. Separate features and target variable.
4. Split the data into training and testing sets with stratification.
5. Preprocess numeric features with a standard scaler.
6. Train a Decision Tree model.
7. Train a Random Forest model.
8. Generate predictions and probability scores.
9. Compare accuracy, precision, recall, F1-score, and ROC-AUC.
10. Visualize confusion matrices and ROC curves.
11. Inspect feature importance from the Random Forest.
12. Select the stronger model based on evaluation results.

## Technologies

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Run the Project

```bash
pip install -r requirements.txt
python src/model.py
```

Generated evaluation plots are saved in `reports/` when the script is run.

For interactive exploration, open `notebooks/predictive_modeling.ipynb` in Jupyter Notebook or VS Code.

## Expected Learning Outcomes

This project provides hands-on practice with supervised learning, classification, train/test evaluation, model comparison, performance visualization, and interpreting predictive models.

## Submission Checklist

- [x] Dataset included
- [x] Data preprocessing included
- [x] Decision Tree implemented
- [x] Random Forest implemented
- [x] Train/test evaluation implemented
- [x] Confusion matrix included
- [x] ROC curve and ROC-AUC included
- [x] Feature importance included
- [x] Model results report included
- [x] Reproducible project structure included

## Author

Saimani94
