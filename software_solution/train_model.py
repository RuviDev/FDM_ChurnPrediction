"""
train_model.py
This script trains an XGBoost‑based churn prediction model on the BankChurners
dataset.  It performs data preprocessing (handling missing values,
standardizing numeric features, and one‑hot encoding categorical features),
splits the data into training and test sets with stratification, fits a tuned
XGBoost classifier inside a scikit‑learn pipeline, evaluates the model using
classification metrics and ROC AUC, and saves the fitted pipeline to
`churn_xgb_model.joblib`.  It also produces a bar chart showing the top 15
feature importances and saves it as `feature_importance.png`.

To run the training:

```bash
python train_model.py
```

The script expects `BankChurners.csv` to be located in the same directory
hierarchy.  The saved model and plot are written into the `software_solution`
folder.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    """Load data, train the model, evaluate, and save artifacts."""
    # Load the dataset.  Drop the helper columns and ID column.
    import pathlib
    # Locate the dataset relative to this script.  The CSV is expected to be in
    # the parent directory of `software_solution`.  Using absolute paths
    # ensures that the script finds the file regardless of the working
    # directory.
    data_path = pathlib.Path(__file__).resolve().parents[1] / "BankChurners.csv"
    df = pd.read_csv(data_path)
    # Drop the last two Naive Bayes helper columns and the customer ID
    df = df.iloc[:, :-2].copy()
    df = df.drop(columns=["CLIENTNUM"], errors="ignore")

    # Create a binary target column: 1 for attrited, 0 for existing customers
    df["Churn"] = df["Attrition_Flag"].map({
        "Attrited Customer": 1,
        "Existing Customer": 0
    })

    # Fill missing values: categoricals with 'Unknown', numerics with median
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].dtype == object:
            df_filled[col] = df_filled[col].fillna("Unknown")
        else:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    # Separate features and target
    X = df_filled.drop(columns=["Attrition_Flag", "Churn"])
    y = df_filled["Churn"]

    # Train/test split with stratification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Identify numeric and categorical columns
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=object).columns.tolist()

    # Define the preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            # Use `sparse=False` for compatibility with sklearn < 1.2.  The
            # newer `sparse_output` argument is not available in older versions.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="passthrough",
    )

    # Compute class weight for the positive class to handle imbalance
    pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Tuned XGBoost parameters (selected via grid search in the notebook)
    xgb_params = {
        "n_estimators": 600,
        "learning_rate": 0.10,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "gamma": 0.0,
        "scale_pos_weight": pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "use_label_encoder": False,
    }

    classifier = XGBClassifier(**xgb_params)

    # Create the full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Existing", "Attrited"], output_dict=True)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Print summary metrics to stdout
    print("Classification report (churn class):")
    print({k: v for k, v in report["Attrited"].items() if k in ("precision", "recall", "f1-score")})
    print(f"ROC AUC: {auc:.4f}")

    # Persist the trained pipeline
    joblib.dump(pipeline, "software_solution/churn_xgb_model.joblib")
    print("Saved model to software_solution/churn_xgb_model.joblib")

    # Derive and plot top feature importances
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_n = 15
    top_features = [feature_names[i] for i in sorted_idx[:top_n]]
    top_importances = importances[sorted_idx[:top_n]]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("software_solution/feature_importance.png")
    print("Saved feature importance plot to software_solution/feature_importance.png")


if __name__ == "__main__":
    main()