"""
predict.py

This script provides a command‑line interface for running churn predictions
using the saved XGBoost pipeline.  It supports two modes:

  * Interactive mode (`--interactive`): prompts the user for values of all
    required features and outputs the churn probability and predicted class.
  * Batch mode (`--input-file`): reads a CSV containing customer records,
    processes it using the same preprocessing as training (missing values and
    one‑hot encoding), and writes a CSV of churn probabilities and labels
    to the specified output path (`--output-file`).

Example usage:

```bash
# Single prediction (interactive prompts)
python predict.py --interactive --model software_solution/churn_xgb_model.joblib

# Batch predictions from CSV
python predict.py --model software_solution/churn_xgb_model.joblib \
    --input-file new_customers.csv --output-file predictions.csv
```
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib


# The exact feature names expected by the model (in order).  These must
# correspond to the raw columns used during training, excluding the ID,
# Attrition_Flag, Churn, and the Naive Bayes helper columns.
EXPECTED_COLS = [
    "Customer_Age",
    "Gender",
    "Dependent_count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]


def load_model(model_path: str):
    """Load a saved sklearn Pipeline from disk."""
    return joblib.load(model_path)


def predict_one(model, data_dict: dict, threshold: float = 0.5) -> dict:
    """Predict churn for a single customer record.

    Parameters
    ----------
    model: Pipeline
        The trained sklearn Pipeline with preprocessor and classifier.
    data_dict: dict
        A mapping from feature names to values.  Missing or unknown keys
        will be filled with NaN and handled by the model's preprocessing.
    threshold: float, optional
        Decision threshold for classifying churn (default 0.5).

    Returns
    -------
    result: dict
        Contains the churn probability, binary label, and threshold used.
    """
    # Build a single‑row DataFrame
    df = pd.DataFrame([data_dict])
    # Ensure all expected columns exist; missing will be NaN
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[EXPECTED_COLS]
    # Fill missing categorical values with 'Unknown'
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    # Predict probability and class
    prob = model.predict_proba(df)[:, 1][0]
    label = int(prob >= threshold)
    return {
        "churn_probability": float(prob),
        "predicted_label": label,
        "threshold_used": float(threshold),
    }


def predict_batch(model, input_csv: str, output_csv: str, threshold: float = 0.5) -> pd.DataFrame:
    """Predict churn for a batch of customers in a CSV file.

    Parameters
    ----------
    model: Pipeline
        The trained sklearn Pipeline.
    input_csv: str
        Path to the CSV file containing raw customer records.
    output_csv: str
        Path where predictions will be written (CSV with probability and label).
    threshold: float
        Decision threshold for classifying churn.

    Returns
    -------
    DataFrame
        A DataFrame with two columns: churn_probability and predicted_label.
    """
    # Read the input file
    raw = pd.read_csv(input_csv)
    # Ensure columns
    for col in EXPECTED_COLS:
        if col not in raw.columns:
            raw[col] = np.nan
    df = raw[EXPECTED_COLS]
    # Fill missing categoricals
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    # Predict probabilities and labels
    probas = model.predict_proba(df)[:, 1]
    labels = (probas >= threshold).astype(int)
    result_df = pd.DataFrame({
        "churn_probability": probas,
        "predicted_label": labels,
    })
    # Write to output file
    result_df.to_csv(output_csv, index=False)
    return result_df


def main() -> None:
    """Parse command line arguments and run predictions."""
    parser = argparse.ArgumentParser(description="Credit Card Churn Prediction CLI")
    parser.add_argument("--model", default="software_solution/churn_xgb_model.joblib",
                        help="Path to the trained model pipeline")
    parser.add_argument("--input-file", dest="input_file", help="CSV file for batch predictions")
    parser.add_argument("--output-file", dest="output_file", default="predictions.csv",
                        help="Where to write batch predictions")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for classifying churn")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    # Load the model
    try:
        model = load_model(args.model)
    except FileNotFoundError:
        print(f"Error: model file {args.model} not found.", file=sys.stderr)
        sys.exit(1)

    # If interactive flag is passed or no input_file provided, run interactive
    if args.interactive or args.input_file is None:
        print("\nEntering interactive mode. Press Enter to leave a field blank.")
        data = {}
        for col in EXPECTED_COLS:
            val = input(f"{col}: ")
            # For numeric columns, try to convert input to float; otherwise use string
            if val == "":
                data[col] = np.nan
            else:
                # Attempt numeric conversion for known numeric features
                if col in [
                    "Customer_Age",
                    "Dependent_count",
                    "Months_on_book",
                    "Total_Relationship_Count",
                    "Months_Inactive_12_mon",
                    "Contacts_Count_12_mon",
                    "Credit_Limit",
                    "Total_Revolving_Bal",
                    "Avg_Open_To_Buy",
                    "Total_Amt_Chng_Q4_Q1",
                    "Total_Trans_Amt",
                    "Total_Trans_Ct",
                    "Total_Ct_Chng_Q4_Q1",
                    "Avg_Utilization_Ratio",
                ]:
                    try:
                        data[col] = float(val)
                    except ValueError:
                        data[col] = np.nan
                else:
                    data[col] = val
        result = predict_one(model, data, threshold=args.threshold)
        print("\nPrediction result:")
        print(f"Churn probability: {result['churn_probability']:.4f}")
        label_text = "Churn" if result['predicted_label'] == 1 else "Not Churn"
        print(f"Predicted class: {label_text}")
    else:
        # Batch mode
        if args.input_file is None:
            parser.error("--input-file is required for batch mode")
        result_df = predict_batch(model, args.input_file, args.output_file, threshold=args.threshold)
        print(f"Processed {len(result_df)} rows. Predictions saved to {args.output_file}.")


if __name__ == "__main__":
    main()