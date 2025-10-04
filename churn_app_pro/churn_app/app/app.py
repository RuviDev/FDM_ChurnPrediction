# app/app.py
import io, json, pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    EXPECTED_COLS, TOP_INPUTS, CAT_OPTIONS, apply_derived_fields,
    reason_codes, recommend_threshold
)

APP_DIR = pathlib.Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
MODEL_PATH = APP_DIR / "churn_model.joblib"
STYLE_PATH = APP_DIR / "styles.css"

st.set_page_config(page_title="Churn Retention Cockpit", page_icon="📉", layout="wide")

# ---------- Styling ----------
if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text()}</style>", unsafe_allow_html=True)

# ---------- Loaders ----------
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["expected_cols"]

@st.cache_data
def load_metrics():
    p = APP_DIR / "metrics.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return {}

def score(df_raw: pd.DataFrame, pipe, expected_cols):
    df = df_raw.copy()
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[expected_cols]
    df = apply_derived_fields(df)
    proba = pipe.predict_proba(df)[:,1]
    return proba

# ---------- Sidebar ----------
def sidebar_controls():
    with st.sidebar:
        st.title("Controls")
        st.caption("ROI-driven thresholding")
        value = st.number_input("Value per retained (USD)", min_value=0.0, value=200.0, step=10.0)
        cost  = st.number_input("Contact cost (USD)",       min_value=0.0, value=2.0,   step=1.0)
        eff   = st.slider("Treatment effectiveness (α)", 0.0, 1.0, 0.30, 0.05)
        st.markdown("---")
        st.caption("Navigation")
        tab = st.radio(
            "Navigation",
            # ["🏠 Home", "📊 EDA", "🧮 Predictor", "🧠 Insights", "📜 Model Card"],
            ["🏠 Home", "📊 EDA", "🧮 Predictor"],
            label_visibility="collapsed",
            key="nav"
        )
    return value, cost, eff, tab

# ---------- Pages ----------
def page_home():
    st.markdown("""
    <div class='hero'>
      <div>
        <h1>Churn Retention Cockpit</h1>
        <p>Batch-first scoring, calibrated probabilities, actionable reason codes, and ROI-driven thresholding—exactly how a bank uses churn models.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Flow", "Batch → Actions")
        st.caption("Upload customer snapshot, get ranked retention queue.")
    with col2:
        st.metric("Reliable risk", "Calibrated")
        st.caption("Isotonic calibration makes probabilities decision-ready.")
    with col3:
        st.metric("Decision", "ROI-optimized")
        st.caption("Set cost & value; we recommend the best threshold.")

    st.markdown("### Quick Start")
    # st.markdown("""
    # 1. Put **BankChurners.csv** in the project root or `app/`.  
    # 2. Run `python train_model.py` to train and save the model + plots.  
    # 3. `streamlit run app/app.py` → go to **Predictor** → **Batch Scoring**.
    # """)

def page_eda():
    st.header("Exploratory Data Analysis")
    st.caption("Upload the original Kaggle CSV. Explore distributions, outliers, churn rate vs features, correlation, and quick auto-insights.")

    up = st.file_uploader("Upload BankChurners.csv", type=["csv"], key="eda_csv")
    if not up:
        st.info("Upload the dataset (original Kaggle CSV).")
        return

    # --- Load & Clean
    df = pd.read_csv(up)
    nb_cols = [c for c in df.columns if c.startswith("Naive_Bayes_Classifier_")]
    df = df.drop(columns=[c for c in ["CLIENTNUM"]+nb_cols if c in df.columns])
    df["Churn"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)

    # Split cols
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    # Remove target from numeric list for plots
    if "Churn" in num_cols:
        num_cols.remove("Churn")

    st.markdown("### Dataset overview")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Rows", f"{len(df):,}")
    with c2: st.metric("Numeric columns", str(len(num_cols)))
    with c3: st.metric("Categorical columns", str(len(cat_cols)))

    st.markdown("#### Class balance")
    bal = df["Churn"].value_counts().rename(index={0:"Existing",1:"Attrited"}).reset_index()
    bal.columns = ["Class","Count"]
    fig_bal = px.bar(bal, x="Class", y="Count", text="Count", title="Churn class balance")
    fig_bal.update_traces(textposition="outside")
    st.plotly_chart(fig_bal, use_container_width=True)

    st.markdown("---")

    # ----------------------
    # Distributions (by churn)
    # ----------------------
    st.markdown("### Distributions by churn")
    default_nums = [c for c in ["Total_Trans_Ct","Total_Trans_Amt","Total_Ct_Chng_Q4_Q1","Total_Amt_Chng_Q4_Q1","Months_Inactive_12_mon"] if c in num_cols][:3]
    sel_num = st.multiselect("Numeric features for distribution", options=num_cols, default=default_nums, key="eda_dist_nums")
    if sel_num:
        for col in sel_num:
            fig = px.histogram(df, x=col, color=df["Churn"].map({0:"Existing",1:"Attrited"}),
                               barmode="overlay", nbins=50,
                               title=f"Distribution of {col} by Churn")
            fig.update_layout(legend_title_text="Class")
            fig.update_traces(opacity=0.65)
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # Outlier diagrams (box/violin)
    # ----------------------
    # st.markdown("### Outliers (box / violin)")
    # col_out1, col_out2 = st.columns(2)
    # with col_out1:
    #     col_box = st.selectbox("Numeric feature (boxplot)", options=num_cols, index=(num_cols.index(default_nums[0]) if default_nums else 0) if num_cols else 0, key="eda_box")
    #     if num_cols:
    #         fig = px.box(df, x=df["Churn"].map({0:"Existing",1:"Attrited"}), y=col_box,
    #                      title=f"Boxplot of {col_box} by Churn")
    #         fig.update_layout(xaxis_title="Class")
    #         st.plotly_chart(fig, use_container_width=True)
    # with col_out2:
    #     col_violin = st.selectbox("Numeric feature (violin)", options=num_cols, index=(num_cols.index(default_nums[1]) if len(default_nums)>1 else 0) if num_cols else 0, key="eda_violin")
    #     if num_cols:
    #         fig = px.violin(df, x=df["Churn"].map({0:"Existing",1:"Attrited"}), y=col_violin, box=True, points="outliers",
    #                         title=f"Violin plot of {col_violin} by Churn")
    #         fig.update_layout(xaxis_title="Class")
    #         st.plotly_chart(fig, use_container_width=True)

    # st.markdown("---")

    # ----------------------
    # Churn rate vs feature (numeric via quantile bins)
    # ----------------------
    st.markdown("### Churn rate vs feature")
    c1, c2 = st.columns(2)
    with c1:
        sel_num_cr = st.selectbox("Numeric feature (binned, churn rate)", options=num_cols, index=(num_cols.index("Total_Trans_Ct") if "Total_Trans_Ct" in num_cols else 0) if num_cols else 0, key="eda_churn_num")
        if num_cols:
            # quantile bins
            try:
                binned = pd.qcut(df[sel_num_cr], q=5, duplicates="drop")
            except Exception:
                # fallback to equal-width if qcut fails
                binned = pd.cut(df[sel_num_cr], bins=5, include_lowest=True)
            tmp = df.groupby(binned, dropna=False)["Churn"].mean().reset_index()
            tmp["bin"] = tmp[sel_num_cr].astype(str)
            fig = px.line(tmp, x="bin", y="Churn", markers=True, title=f"Churn rate across {sel_num_cr} bins")
            fig.update_layout(xaxis_title=f"{sel_num_cr} bins", yaxis_title="Churn rate")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Categorical churn rate
        cat_opts = [c for c in cat_cols if c not in ["Attrition_Flag"]]
        sel_cat_cr = st.selectbox("Categorical feature (churn rate)", options=cat_opts, index=(cat_opts.index("Card_Category") if "Card_Category" in cat_opts else 0) if cat_opts else 0, key="eda_churn_cat")
        if cat_opts:
            tmp = df.groupby(sel_cat_cr)["Churn"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(tmp, x=sel_cat_cr, y="Churn", title=f"Churn rate by {sel_cat_cr}")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ----------------------
    # Correlation matrix (numeric features)
    # ----------------------
    # st.markdown("### Correlation matrix (numeric features)")
    # num_for_corr = ["Churn"] + num_cols
    # corr = df[num_for_corr].corr(numeric_only=True)
    # fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower",
    #                 title="Correlation heatmap (including Churn)")
    # st.plotly_chart(fig, use_container_width=True)

    # st.markdown("---")

    # ----------------------
    # Auto-Insights (quick highlights)
    # ----------------------
    with st.expander("Auto-Insights (top signals detected)", expanded=True):
        bullets = []

        # Numeric: mean diff between churned vs existing
        diffs = []
        for c in num_cols:
            g = df.groupby("Churn")[c].mean()
            if 0 in g.index and 1 in g.index:
                diffs.append((c, float(g[1] - g[0])))
        diffs = sorted(diffs, key=lambda x: abs(x[1]), reverse=True)[:5]
        if diffs:
            bullets.append("**Numeric drivers (mean difference: churn − existing):** " +
                           ", ".join([f"{c} ({d:+.2f})" for c,d in diffs]))

        # Categorical: top categories by churn rate
        cat_msg = []
        for c in [x for x in cat_cols if x not in ["Attrition_Flag"]]:
            rates = df.groupby(c)["Churn"].mean().sort_values(ascending=False)
            top = rates.head(2)
            if len(top) > 0:
                top_str = "; ".join([f"{idx}: {val:.2%}" for idx,val in top.items()])
                cat_msg.append(f"{c} → {top_str}")
        if cat_msg:
            bullets.append("**Categorical hot spots (highest churn rates):** " + " | ".join(cat_msg[:3]))

        if not bullets:
            st.write("No strong patterns detected automatically.")
        else:
            for b in bullets:
                st.markdown(f"- {b}")

def page_predictor(pipe, expected_cols, value, cost, eff):
    st.header("Predictor")
    tab1, tab2 = st.tabs(["📦 Batch Scoring (recommended)", "👤 Single Customer (what‑if)"])

    with tab1:
        st.write("Upload a CSV. Extra columns are ignored; derived fields are auto-computed.")
        up = st.file_uploader("Upload CSV", type=["csv"], key="batch")
        if up:
            df_in = pd.read_csv(up)
            df_in = apply_derived_fields(df_in)
            proba = score(df_in, pipe, expected_cols)
            t_star, ev_star, ts, evs = recommend_threshold(proba, value, cost, eff)
            st.success(f"Recommended threshold: **{t_star:.2f}**  |  Max Expected Value: **${ev_star:,.0f}**")
            threshold = st.slider("Decision threshold", 0.1, 0.9, float(t_star), 0.01, key="thr")
            label = (proba >= threshold).astype(int)
            enriched = df_in.copy()
            enriched["churn_probability"] = proba
            enriched["predicted_label"] = label
            reasons_list, actions = [], []
            for i in range(len(enriched)):
                info = reason_codes(enriched.iloc[i], proba[i])
                reasons_list.append(" • ".join(info["reasons"]))
                actions.append(info["action"])
            enriched["top_reasons"] = reasons_list
            enriched["recommended_action"] = actions
            id_cols = [c for c in enriched.columns if c.lower() in ("clientnum","customer_id","id")]
            show_cols = id_cols + ["churn_probability","predicted_label","top_reasons","recommended_action"]
            st.markdown("#### Ranked Retention Queue")
            st.dataframe(enriched.sort_values("churn_probability", ascending=False)[show_cols].head(50), use_container_width=True)
            st.download_button("Download full results CSV", data=enriched.to_csv(index=False).encode("utf-8"),
                               file_name="retention_queue.csv", mime="text/csv")
        else:
            st.info("Awaiting CSV upload…")

    with tab2:
        st.caption("Minimal inputs; derived fields are computed for you.")
        c1, c2 = st.columns(2)
        with c1:
            total_trans_ct      = st.number_input("Total_Trans_Ct", min_value=0, value=12, step=1)
            total_ct_chg_q4_q1  = st.number_input("Total_Ct_Chng_Q4_Q1", min_value=0.0, max_value=5.0, value=0.45, step=0.05)
            total_amt_chg_q4_q1 = st.number_input("Total_Amt_Chng_Q4_Q1", min_value=0.0, max_value=5.0, value=0.50, step=0.05)
            months_inactive     = st.number_input("Months_Inactive_12_mon", min_value=0, max_value=12, value=5, step=1)
            contacts_12_mon     = st.number_input("Contacts_Count_12_mon", min_value=0, max_value=20, value=5, step=1)
        with c2:
            rel_count  = st.number_input("Total_Relationship_Count", min_value=0, max_value=10, value=1, step=1)
            credit_lim = st.number_input("Credit_Limit", min_value=200.0, max_value=100000.0, value=2000.0, step=100.0)
            revol_bal  = st.number_input("Total_Revolving_Bal", min_value=0.0, max_value=100000.0, value=500.0, step=50.0)
            income     = st.selectbox("Income_Category", CAT_OPTIONS["Income_Category"], index=1)
            card       = st.selectbox("Card_Category", CAT_OPTIONS["Card_Category"], index=0)

        row = {c: np.nan for c in EXPECTED_COLS}
        row.update({
            "Total_Trans_Ct": total_trans_ct,
            "Total_Ct_Chng_Q4_Q1": total_ct_chg_q4_q1,
            "Total_Amt_Chng_Q4_Q1": total_amt_chg_q4_q1,
            "Months_Inactive_12_mon": months_inactive,
            "Contacts_Count_12_mon": contacts_12_mon,
            "Total_Relationship_Count": rel_count,
            "Credit_Limit": credit_lim,
            "Total_Revolving_Bal": revol_bal,
            "Income_Category": income,
            "Card_Category": card
        })
        df1 = pd.DataFrame([row])
        df1 = apply_derived_fields(df1)
        proba = score(df1, pipe, expected_cols)[0]
        info  = reason_codes(df1.iloc[0], proba)
        st.metric("Churn probability", f"{proba:.3f}")
        st.write("**Top reasons**:", " | ".join(info["reasons"]))
        st.write("**Recommended action**:", info["action"])

        st.write("— What‑if: increase Total_Trans_Ct —")
        new_ct = st.slider("Proposed Total_Trans_Ct", 0, 100, int(total_trans_ct), 1)
        df2 = df1.copy(); df2.loc[0, "Total_Trans_Ct"] = new_ct
        proba2 = score(df2, pipe, expected_cols)[0]
        delta = proba2 - proba
        st.metric("New probability", f"{proba2:.3f}", delta=f"{delta:+.3f}")

def page_insights():
    st.header("Insights")
    st.caption("Holdout metrics and diagnostic plots from training.")
    cols = st.columns(2)
    imgs = [("roc.png","ROC Curve"), ("pr.png","Precision-Recall"), ("calibration.png","Calibration (Reliability)"), ("feature_importance.png","Top Feature Importances")]
    for i,(fname, title) in enumerate(imgs):
        with cols[i%2]:
            p = ASSETS / fname
            if p.exists():
                st.image(str(p), caption=title, use_column_width=True)
            else:
                st.info(f"{fname} not found. Run train_model.py to regenerate.")
    metrics_path = APP_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f: metrics = json.load(f)
        st.markdown("### Summary metrics")
        st.json(metrics)

def page_model_card():
    st.header("Model Card")
    md = (APP_DIR / "model_card.md")
    if md.exists():
        st.markdown(md.read_text())
    else:
        st.info("model_card.md not found. Create one describing data, preprocessing, model, metrics, intended use, limitations, fairness, and monitoring.")

def main():
    value, cost, eff, tab = sidebar_controls()
    st.markdown("<div class='topbar'>Bank Churn Prediction • Professional Demo</div>", unsafe_allow_html=True)

    if tab == "🏠 Home":
        page_home(); return
    if tab == "📊 EDA":
        page_eda(); return
    # if tab == "🧠 Insights":
    #     page_insights(); return
    # if tab == "📜 Model Card":
    #     page_model_card(); return

    # Predictor needs the model
    try:
        pipe, expected_cols = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}. Please run `python train_model.py` first.")
        return

    if tab == "🧮 Predictor":
        page_predictor(pipe, expected_cols, value, cost, eff)

if __name__ == "__main__":
    main()
