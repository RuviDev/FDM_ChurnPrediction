# app/utils.py
import numpy as np
import pandas as pd

EXPECTED_COLS = [
    'Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status',
    'Income_Category','Card_Category','Months_on_book','Total_Relationship_Count',
    'Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit',
    'Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio'
]

TOP_INPUTS = [
    'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Total_Amt_Chng_Q4_Q1',
    'Months_Inactive_12_mon','Contacts_Count_12_mon','Total_Relationship_Count',
    'Credit_Limit','Total_Revolving_Bal','Income_Category','Card_Category'
]

CAT_OPTIONS = {
    "Income_Category": ["Unknown","Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +"],
    "Card_Category": ["Blue","Silver","Gold","Platinum"]
}

def apply_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Credit_Limit" in df and "Total_Revolving_Bal" in df:
        df["Avg_Open_To_Buy"] = df["Credit_Limit"] - df["Total_Revolving_Bal"]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = df["Total_Revolving_Bal"] / df["Credit_Limit"]
            df["Avg_Utilization_Ratio"] = ratio.replace([np.inf, -np.inf], 0).fillna(0)
    return df

def reason_codes(row: pd.Series, proba: float) -> dict:
    reasons = []
    if row.get("Total_Trans_Ct", 999) <= 20:
        reasons.append("Very low transaction count")
    if row.get("Total_Ct_Chng_Q4_Q1", 1.0) < 0.7:
        reasons.append("Drop in transaction frequency (Q4/Q1)")
    if row.get("Total_Amt_Chng_Q4_Q1", 1.0) < 0.7:
        reasons.append("Drop in spend amount (Q4/Q1)")
    if row.get("Months_Inactive_12_mon", 0) >= 4:
        reasons.append("High inactivity (12 months)")
    if row.get("Contacts_Count_12_mon", 0) >= 5:
        reasons.append("High customer support contacts")
    if row.get("Total_Relationship_Count", 99) <= 2:
        reasons.append("Weak relationship (few products)")
    if row.get("Avg_Utilization_Ratio", 0) < 0.05:
        reasons.append("Very low utilization (possible disengagement)")
    if row.get("Income_Category","Unknown") in ["Unknown","Less than $40K"] and row.get("Card_Category","Blue")=="Blue":
        reasons.append("Low-income Blue segment with low engagement")

    action = "Re-activation: cashback on next 5 transactions"
    if row.get("Months_Inactive_12_mon",0) >= 4:
        action = "Inactivity nudge + app push re-engagement"
    if row.get("Total_Relationship_Count",99) <= 2:
        action = "Cross-sell bundle (no-fee savings + rewards)"
    return {"reasons": reasons[:3], "action": action}

def expected_value(prob: np.ndarray, threshold: float, value_per_retained: float, contact_cost: float, effectiveness: float) -> float:
    contact = (prob >= threshold).astype(int)
    gains = effectiveness * prob * value_per_retained - contact_cost
    return float((gains * contact).sum())

def recommend_threshold(prob: np.ndarray, value_per_retained: float, contact_cost: float, effectiveness: float):
    thresholds = np.linspace(0.1, 0.9, 81)
    evs = [expected_value(prob, t, value_per_retained, contact_cost, effectiveness) for t in thresholds]
    best_idx = int(np.argmax(evs))
    return float(thresholds[best_idx]), float(evs[best_idx]), thresholds, evs
