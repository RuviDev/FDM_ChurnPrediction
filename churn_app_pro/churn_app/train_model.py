"""
Train a calibrated XGBoost churn model (BankChurners).
- Robust column handling (drops CLIENTNUM + Naive_Bayes_* helper cols)
- Pipeline with imputers + scaler + OHE(handle_unknown='ignore')
- Isotonic calibration for trustworthy probabilities
- Saves: app/churn_model.joblib, app/assets/{roc,pr,calibration,feature_importance}.png, app/metrics.json
"""
import json, pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, brier_score_loss
)
from xgboost import XGBClassifier

def load_dataset():
    here = pathlib.Path(__file__).resolve().parent
    candidates = [
        here / "BankChurners.csv",
        here / "app" / "BankChurners.csv"
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("BankChurners.csv not found. Place it in project root or /app.")

def prepare(df: pd.DataFrame):
    nb_cols = [c for c in df.columns if c.startswith("Naive_Bayes_Classifier_")]
    drop_cols = ["CLIENTNUM"] + nb_cols
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df["Churn"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    expected_cols = [
        'Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status',
        'Income_Category','Card_Category','Months_on_book','Total_Relationship_Count',
        'Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit',
        'Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio'
    ]
    X = df[expected_cols].copy()
    y = df["Churn"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, expected_cols

def build_pipeline(numeric_features, categorical_features):
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0
    )
    calibrated = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=5)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", calibrated)])
    return pipe

def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test, expected_cols = prepare(df)

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / max(pos, 1)

    pipe = build_pipeline(numeric_features, categorical_features)
    # set scale_pos_weight on underlying XGB estimator
    pipe.named_steps["classifier"].estimator.set_params(scale_pos_weight=spw)

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    # --- Evaluate the final model ---
    print("\n--- Final Model Evaluation ---")
    y_pred_proba_final = pipe.predict_proba(X_test)[:, 1]
    final_auc = roc_auc_score(y_test, y_pred_proba_final)

    y_pred_final = pipe.predict(X_test)
    final_report = classification_report(y_test, y_pred_final, target_names=['Existing (0)', 'Attrited (1)'], output_dict=True)

    print(f"Final Calibrated XGB F1-Score (Churn): {final_report['Attrited (1)']['f1-score']:.4f}")
    print(f"Final Calibrated XGB AUC: {final_auc:.4f}")

    # -------------------------------------

    print("Classification report (churn class=1):")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("PR  AUC:", average_precision_score(y_test, proba))
    print("Brier  :", brier_score_loss(y_test, proba))

    out_dir = pathlib.Path(__file__).resolve().parent / "app"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "churn_model.joblib"
    import joblib as _joblib
    _joblib.dump({"pipeline": pipe, "expected_cols": expected_cols}, model_path)
    print(f"Saved model to {model_path}")

    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, proba)
    import matplotlib
    matplotlib.use("Agg")
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.title("ROC"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(True); plt.tight_layout(); plt.savefig(assets / "roc.png"); plt.close()

    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure(); plt.plot(rec, prec); plt.title("Precision-Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(True); plt.tight_layout(); plt.savefig(assets / "pr.png"); plt.close()

    bins = np.linspace(0,1,11)
    digitized = np.digitize(proba, bins) - 1
    bin_centers, bin_conf, bin_frac = [], [], []
    for b in range(10):
        mask = digitized==b
        if mask.any():
            bin_centers.append( (bins[b]+bins[b+1])/2 )
            bin_conf.append( proba[mask].mean() )
            bin_frac.append( y_test[mask].mean() )
    plt.figure(); plt.plot(bin_conf, bin_frac, 'o-'); plt.plot([0,1],[0,1],'--'); plt.title("Calibration (Reliability)"); plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.grid(True); plt.tight_layout(); plt.savefig(assets / "calibration.png"); plt.close()

    # Feature importance via a non-calibrated clone
    from copy import deepcopy
    xgb_imp = deepcopy(pipe)
    xgb_imp.steps[-1] = ('classifier', XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=42, n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
        scale_pos_weight=spw
    ))
    xgb_imp.fit(X_train, y_train)
    pre = xgb_imp.named_steps["preprocessor"]
    feat_names = pre.get_feature_names_out()
    importances = xgb_imp.named_steps["classifier"].feature_importances_
    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8,6))
    plt.barh([feat_names[i] for i in order][::-1], importances[order][::-1])
    plt.title("Top 20 Feature Importances (XGB)")
    plt.tight_layout(); plt.savefig(assets / "feature_importance.png"); plt.close()

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
