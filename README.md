# Churn Retention Cockpit (Bank Churn Prediction App)

Interactive Streamlit app for **credit-card customer churn prediction and retention planning**, built on the public **Kaggle BankChurners** dataset.

## 🎥 Demo

▶ [Watch the demo video on YouTube](https://youtu.be/oMLMxdxn1bM)

The app lets CRM / product / analytics teams:

- Explore churn patterns and segment behavior
- Score customers with a **calibrated XGBoost model**
- Build a **ROI-driven retention queue** with simple financial assumptions
- Run quick **“what-if” simulations** for individual customers

---

## Features

### 🏠 Home

- High-level value proposition and quick tour of the app.
- Embedded **model card** (`app/model_card.md`) describing:
  - Data source and preprocessing
  - Intended use and limitations
  - Calibration and evaluation metrics
  - Fairness and governance considerations

### 📊 EDA – Manager’s Churn Insights

- Upload a BankChurners-style CSV or click **Use sample data**.
- Automatic label creation: `Churn = 1` if `Attrition_Flag == "Attrited Customer"`, else `0`.
- Cleans helper columns:
  - Drops `CLIENTNUM` and `Naive_Bayes_Classifier_*`.
- Overview KPIs:
  - Total customers, churn rate, retained customers.
- Visuals (Plotly):
  - Class balance (existing vs attrited).
  - Distributions of key numeric features, split by churn.
  - Churn rate across binned numeric features.
  - Churn rate by categorical segments (income, card category, etc.).
- Textual bullets summarizing **top churn drivers** for numeric + categorical features.

### 🧮 Predictor

Two workflows in one page:

#### 1) 📦 Batch Scoring (recommended)

- Upload a CSV of **current customers**.
- The app:
  - Applies `apply_derived_fields(...)` to align with the model’s expected columns.
  - Scores each customer with a calibrated XGBoost pipeline.
- ROI sidebar controls:
  - **Average value per retained customer ($)**
  - **Cost per retention contact ($)**
  - **Retention effectiveness** (share of contacted churners who actually stay)
- Uses `recommend_threshold(...)` to search thresholds `t ∈ [0.1, 0.9]` and choose the one that **maximizes expected value**:

  > EV = effectiveness × P(churn) × value_per_retained − contact_cost

- Produces a **retention queue** dataframe with:
  - `churn_probability`
  - `predicted_label` (“Attrited” / “Existing”)
  - `top_reasons` (simple rule-based reason codes per customer)

You can download or further slice/segment this table directly in the Streamlit UI.

#### 2) 👤 Single Customer (what-if)

- Interactive form to capture the most important features:

  - `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Total_Amt_Chng_Q4_Q1`
  - `Months_Inactive_12_mon`, `Contacts_Count_12_mon`
  - `Total_Relationship_Count`, `Credit_Limit`, `Total_Revolving_Bal`
  - `Income_Category`, `Card_Category`, `Customer_Age`, etc.

- The app builds a full feature row, runs it through the pipeline and shows:
  - Churn probability
  - Top 2–3 **reason codes** from `reason_codes(...)` (e.g. low transactions, inactive, high contact rate).
- **What-if slider**:
  - Simulate increasing `Total_Trans_Ct`.
  - See the new predicted churn probability and the delta vs the original.

---

## Model & Training

Training code lives in `churn_app/train_model.py`. It:

1. **Loads data** with `load_dataset()`  
   Looks for:

   - `churn_app/BankChurners.csv`
   - `churn_app/app/BankChurners.csv`

2. **Prepares features** with `prepare(...)`  
   - Drops `CLIENTNUM` and `Naive_Bayes_Classifier_*` columns.
   - Creates binary `Churn` from `Attrition_Flag`.
   - Uses this feature set:

     ```text
     Customer_Age
     Gender
     Dependent_count
     Education_Level
     Marital_Status
     Income_Category
     Card_Category
     Months_on_book
     Total_Relationship_Count
     Months_Inactive_12_mon
     Contacts_Count_12_mon
     Credit_Limit
     Total_Revolving_Bal
     Avg_Open_To_Buy
     Total_Amt_Chng_Q4_Q1
     Total_Trans_Amt
     Total_Trans_Ct
     Total_Ct_Chng_Q4_Q1
     Avg_Utilization_Ratio
     ```

   - 70/30 **stratified** train/test split.

3. **Builds a pipeline** with `build_pipeline(...)`  
   - Numeric features: `SimpleImputer(strategy="median")` + `StandardScaler`.
   - Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
   - Classifier: `XGBClassifier` with sensible defaults (`tree_method="hist"`, regularization, etc.).
   - Wraps with **`CalibratedClassifierCV` (isotonic)** for probability calibration.

4. **Evaluates & saves artifacts**

   - Metrics (hold-out):

     - ROC AUC ≈ **0.9933**
     - PR AUC  ≈ **0.9675**

     Exact values + full classification report are stored in `churn_app/app/metrics.json`.

   - Plots written to `churn_app/app/assets/`:
     - `roc.png`
     - `pr.png`
     - `calibration.png`
     - `feature_importance.png`
   - Model bundle:
     - `churn_app/app/churn_model.joblib` (dict with `"pipeline"` and `"expected_cols"`).

---

## Project Structure

After unpacking:

```text
churn_app_pro/
├── churn_app/
│   ├── app/
│   │   ├── app.py              # Streamlit application
│   │   ├── utils.py            # Feature lists, reason codes, ROI helpers
│   │   ├── styles.css          # Custom styling for Streamlit app
│   │   ├── BankChurners.csv    # Sample training / EDA dataset
│   │   ├── churn_model.joblib  # Trained, calibrated pipeline
│   │   ├── metrics.json        # Saved evaluation metrics
│   │   ├── model_card.md       # Model card rendered on Home
│   │   └── assets/
│   ├── sampleDataForTesting/
│   │   ├── eda.csv             # Small sample for EDA
│   │   └── sampleBatch20.csv   # Sample batch for predictions
│   ├── requirements.txt        # App + training dependencies
│   └── train_model.py          # Training script
└── colab_codes/
    └── FDM_calibratedModel.ipynb  # Notebook for experimentation
