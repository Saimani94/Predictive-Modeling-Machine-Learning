from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_purchase_data.csv"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = REPORT_DIR / "plots"
REPORT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")

    expected = {
        "age",
        "annual_income",
        "years_experience",
        "website_visits",
        "avg_session_minutes",
        "purchased",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()
    numeric_cols = [
        "age",
        "annual_income",
        "years_experience",
        "website_visits",
        "avg_session_minutes",
    ]

    # Basic quality checks and deterministic imputation.
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["purchased"] = pd.to_numeric(df["purchased"], errors="coerce")
    df = df.dropna(subset=["purchased"])
    df["purchased"] = df["purchased"].astype(int)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[numeric_cols]
    y = df["purchased"]
    return df, X, y


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return metrics, predictions, probabilities


def save_confusion_matrix(name, y_test, predictions):
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"],
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(PLOT_DIR / filename, dpi=160)
    plt.close()


def save_roc_curve(results):
    plt.figure(figsize=(7, 5))
    for result in results:
        fpr, tpr, _ = roc_curve(result["y_test"], result["probabilities"])
        plt.plot(fpr, tpr, label=f'{result["name"]} (AUC={result["metrics"]["roc_auc"]:.3f})')

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve_comparison.png", dpi=160)
    plt.close()


def save_feature_importance(model, feature_names):
    classifier = model.named_steps["classifier"]
    importance = pd.Series(classifier.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance.values, y=importance.index)
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "random_forest_feature_importance.png", dpi=160)
    plt.close()
    return importance


def write_report(metrics_df, best_model, importance, row_count, class_balance):
    lines = [
        "# Predictive Modeling Results",
        "",
        "## Dataset summary",
        f"- Records used after cleaning: **{row_count}**",
        f"- No-purchase class (0): **{class_balance.get(0, 0)}**",
        f"- Purchase class (1): **{class_balance.get(1, 0)}**",
        "",
        "## Model comparison",
        "",
        metrics_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"## Selected model",
        f"**{best_model}** achieved the strongest ROC-AUC among the evaluated models on the held-out test set.",
        "",
        "## Feature importance",
        "",
    ]
    for feature, value in importance.items():
        lines.append(f"- `{feature}`: {value:.3f}")

    lines += [
        "",
        "## Interpretation",
        "",
        "The model comparison demonstrates a complete supervised-learning workflow: preprocessing, train/test splitting, model training, probability-based evaluation, and visual diagnostics.",
        "",
        "## Limitations",
        "",
        "This is a small illustrative dataset. Results are intended for learning and portfolio demonstration rather than production decision-making.",
    ]
    (REPORT_DIR / "model_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    raw_df = load_data()
    df, X, y = preprocess(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200, max_depth=6, random_state=42
                    ),
                ),
            ]
        ),
    }

    metrics = []
    curve_results = []
    fitted_models = {}

    for name, model in models.items():
        model_metrics, predictions, probabilities = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        metrics.append(model_metrics)
        fitted_models[name] = model
        curve_results.append(
            {
                "name": name,
                "metrics": model_metrics,
                "probabilities": probabilities,
                "y_test": y_test,
            }
        )
        save_confusion_matrix(name, y_test, predictions)

        print(f"\n{name}")
        print(classification_report(y_test, predictions, zero_division=0))

    metrics_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    best_model_name = metrics_df.iloc[0]["model"]
    importance = save_feature_importance(fitted_models["Random Forest"], X.columns)
    save_roc_curve(curve_results)

    write_report(
        metrics_df,
        best_model_name,
        importance,
        len(df),
        y.value_counts().sort_index().to_dict(),
    )

    print("\nModel comparison:")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest model by ROC-AUC: {best_model_name}")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
