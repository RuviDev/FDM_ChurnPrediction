# Credit Card Customer Churn Prediction Software

This directory contains a complete software solution for predicting credit card
customer churn using a machine‑learning model trained on the publicly
available **BankChurners** dataset from Kaggle.  It encompasses data
preprocessing, model training, evaluation, and both command‑line and
web‑based interfaces for scoring new customers.

## Components

| File | Description |
| --- | --- |
| `train_model.py` | Loads the Kaggle **BankChurners** dataset, performs preprocessing (handling missing values, standardising numeric columns and one‑hot encoding categorical columns), trains a tuned XGBoost classifier using a scikit‑learn pipeline, evaluates the model on a stratified 70/30 split, and saves the fitted pipeline to `churn_xgb_model.joblib`.  It also generates a bar chart showing the top 15 feature importances and saves it as `feature_importance.png`. |
| `predict.py` | Command‑line tool for running predictions using the saved pipeline.  Supports interactive single‑record input and batch predictions from a CSV file.  Outputs a CSV containing churn probabilities and predicted labels. |
| `app.py` | Optional Streamlit application that provides a user‑friendly web interface.  Users can enter details for a single customer or upload a CSV file for batch scoring.  The sidebar lets users adjust the decision threshold and displays the top features.  Note: running this app requires installing `streamlit` in your environment (`pip install streamlit`). |
| `churn_xgb_model.joblib` | (Generated after running `train_model.py`) The fitted scikit‑learn pipeline combining preprocessing and the tuned XGBoost classifier. |
| `feature_importance.png` | (Generated after running `train_model.py`) Bar chart of the top 15 feature importances for the tuned XGBoost model. |

## Usage

### 1. Train the model

Make sure you have Python 3.8+ installed and the required packages (`pandas`,
`numpy`, `scikit‑learn`, `xgboost`, `seaborn`, `matplotlib`, and `joblib`).

Run the training script from the project root:

```bash
python software_solution/train_model.py
```

This will output evaluation metrics to the console, save the fitted model to
`software_solution/churn_xgb_model.joblib` and create a feature importance plot
as `software_solution/feature_importance.png`.

### 2. Make predictions

#### Interactive mode

Run the prediction script without specifying an input file.  You will be
prompted to enter each feature.  Leave a field blank to use the default
(`NaN`, which will be handled by the model's preprocessing):

```bash
python software_solution/predict.py --interactive --model software_solution/churn_xgb_model.joblib
```

#### Batch mode

Provide a CSV file containing the raw feature columns (see
`predict.py` for the list of `EXPECTED_COLS`) and an output path.  The tool
will process the file and write a new CSV containing churn probabilities and
binary labels:

```bash
python software_solution/predict.py --model software_solution/churn_xgb_model.joblib \
    --input-file path/to/new_customers.csv --output-file path/to/predictions.csv --threshold 0.5
```

### 3. Run the Streamlit app (optional)

If you wish to provide a graphical interface, install Streamlit and run:

```bash
pip install streamlit
streamlit run software_solution/app.py
```

The app offers two modes of operation (single and batch) and allows you to
adjust the classification threshold via the sidebar.  Streamlit provides an
easy way to build interactive ML apps without managing backend routes or
HTML.  As explained in a Docker tutorial on deploying churn prediction apps,
Streamlit’s minimal API lets you build UI components in pure Python【643571009368409†L139-L151】.
Furthermore, the sidebar pattern adopted here is inspired by the same
tutorial, which demonstrates switching between online and batch scoring
modes to enhance user experience【643571009368409†L360-L369】.

## Model details

The underlying model is a tuned **XGBoost** classifier trained on the
BankChurners dataset.  It uses `scale_pos_weight` to handle the class
imbalance (# of retained customers / # of churned customers) and hyperparameters
selected via cross‑validated grid search.  Preprocessing is performed by a
`ColumnTransformer` that applies `StandardScaler` to numeric features and
`OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to categorical
features, ensuring that unseen categories at inference time do not cause
errors.  The output pipeline can be loaded with `joblib.load` and used to
make predictions on raw customer data.

## License

This project is provided for educational purposes.  The BankChurners dataset
is publicly available under Kaggle’s Terms of Service.