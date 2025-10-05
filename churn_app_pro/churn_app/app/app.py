import io, json, pathlib
import joblib
import base64
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

# --- add near your APP_DIR/ASSETS constants ---
BRAND_NAME = "Metrics"

# Resolve images regardless of where you put them (assets/, app/ root, or /mnt/data/)
LOGO_CANDIDATES = [ASSETS / "CompanyLogo.png", APP_DIR / "CompanyLogo.png", pathlib.Path("/mnt/data/CompanyLogo.png")]
WORDMARK_CANDIDATES = [ASSETS / "CompanyName.png", APP_DIR / "CompanyName.png", pathlib.Path("/mnt/data/CompanyName.png")]
LOGO_PATH = next((p for p in LOGO_CANDIDATES if p.exists()), None)
WORDMARK_PATH = next((p for p in WORDMARK_CANDIDATES if p.exists()), None)

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
        # --- BRAND HEADER (icon + name side-by-side) ---
        st.markdown(
            """
            <style>
              .brand { display:flex; align-items:top; gap:.5rem; margin-bottom:.75rem; margin-top:-3rem; }
              .brand img.brand-logo { height:39px; width:38px; }
              .brand img.wordmark   { height:28px; width:auto; padding-top:6px; }
              .brand .brand-name    { font-weight:700; font-size:1.1rem; }
            </style>
            """,
            unsafe_allow_html=True
        )

        if LOGO_PATH:
            logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
            if WORDMARK_PATH:
                wordmark_b64 = base64.b64encode(WORDMARK_PATH.read_bytes()).decode()
                st.markdown(
                    f"<div class='brand'>"
                    f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='logo'/>"
                    f"<img class='wordmark'   src='data:image/png;base64,{wordmark_b64}' alt='Metrics'/>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                # Fallback: use text name if you prefer not to use the wordmark image
                st.markdown(
                    f"<div class='brand'>"
                    f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='logo'/>"
                    f"<span class='brand-name'>{BRAND_NAME}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"<div class='brand'><span class='brand-name'>{BRAND_NAME}</span></div>", unsafe_allow_html=True)

        # --- your existing sidebar UI starts here ---
        # st.title("Controls")
        # st.caption("ROI-driven thresholding")
        # value = st.number_input("Value per retained (USD)", min_value=0.0, value=200.0, step=10.0)
        # cost  = st.number_input("Contact cost (USD)",       min_value=0.0, value=2.0,   step=1.0)
        # eff   = st.slider("Treatment effectiveness (α)", 0.0, 1.0, 0.30, 0.05)
        # st.markdown("---")
        st.caption("Navigation")
        tab = st.radio(
            "Navigation",
            ["🏠 Home", "📊 EDA", "🧮 Predictor"],
            label_visibility="collapsed",
            key="nav"
        )
    return tab

def sidebar_nav():
    # id, label, icon
    NAV = [
        ("home",       "Home",       "🏠"),
        ("eda",        "EDA",        "📊"),
        ("predictor",  "Predictor",  "🧮"),
    ]

    qp = st.query_params
    current = qp.get("tab", NAV[0][0])
    valid_ids = {k for k, _, _ in NAV}
    if current not in valid_ids:
        current = NAV[0][0]
    if st.query_params.get("tab") != current:
        st.query_params["tab"] = current

    st.markdown(
        """
        <style>
          .nav { display:flex; flex-direction:column; gap:6px; margin-top:.25rem; }
          .nav a { display:flex; align-items:center; gap:.6rem; padding:10px 12px;
                   border-radius:10px; text-decoration:none; font-weight:600;
                   border:1px solid rgba(255,255,255,.08); color:inherit; transition:all .12s; }
          .nav a:hover { background:rgba(255,255,255,.06); transform:translateX(1px); }
          .nav a.active { background:rgba(255,255,255,.10); border-color:rgba(255,255,255,.18);
                          box-shadow: inset 2px 0 0 0 #22c55e; }
          .nav .icon { width:1.1rem; text-align:center; }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown(
            """
            <style>
              .brand { display:flex; align-items:top; gap:.5rem; margin-bottom:.75rem; margin-top:-3rem; }
              .brand img.brand-logo { height:39px; width:38px; }
              .brand img.wordmark   { height:32px; width:auto; padding-top:6px; }
              .brand .brand-name    { font-weight:700; font-size:1.1rem; }
            </style>
            """,
            unsafe_allow_html=True
        )

        if LOGO_PATH:
            logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
            if WORDMARK_PATH:
                wordmark_b64 = base64.b64encode(WORDMARK_PATH.read_bytes()).decode()
                st.markdown(
                    f"<div class='brand'>"
                    f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='logo'/>"
                    f"<img class='wordmark'   src='data:image/png;base64,{wordmark_b64}' alt='Metrics'/>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                # Fallback: use text name if you prefer not to use the wordmark image
                st.markdown(
                    f"<div class='brand'>"
                    f"<img class='brand-logo' src='data:image/png;base64,{logo_b64}' alt='logo'/>"
                    f"<span class='brand-name'>{BRAND_NAME}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"<div class='brand'><span class='brand-name'>{BRAND_NAME}</span></div>", unsafe_allow_html=True)

        st.markdown("#### Navigation")
        links = []
        for k, label, icon in NAV:
            active = "active" if k == current else ""
            # NOTE: target="_self" keeps it in the same tab
            links.append(
                f'<a class="{active}" href="?tab={k}" target="_self">'
                f'<span class="icon">{icon}</span><span>{label}</span></a>'
            )
        st.markdown(f'<div class="nav">{"".join(links)}</div>', unsafe_allow_html=True)

    label_map = {k: f"{icon} {label}" for k, label, icon in NAV}
    return label_map[current]


# ---------- Pages ----------
def page_home():
    # Use a visually appealing, centered layout for the main title and value proposition
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🏦 Churn Retention Cockpit</h1>
            <p>A smart dashboard to help you <strong>predict which customers are likely to leave</strong>, 
            understand <em>why</em>, and decide the most profitable way to retain them.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- "How It Works" Section ---
    st.header("How It Works")

    # Using gap="large" gives the columns a bit more space
    col1, col2, col3 = st.columns(3, gap="large") 

    with col1:
        st.subheader("Spot At-Risk Clients")
        st.info(
            "Upload your customer list. Our AI model will instantly score each customer, creating a "
            "prioritized list of who is most likely to leave the bank."
        )

    with col2:
        st.subheader("Understand the Risk")
        st.info(
            "For every at-risk customer, the tool provides the top reasons for their churn risk "
            "and suggests a clear, recommended action to prevent churn."
        )

    with col3:
        st.subheader("Make Smart Decisions")
        st.info(
            "Simulate the impact of your actions. Use our 'What-If' analysis and ROI calculator "
            "to find the most profitable strategy for your retention campaigns."
        )

    st.markdown("---")

    # --- "Get Started" Section ---
    st.header("Get Started")
    st.success(
        "Navigate using the sidebar on the left to explore the cockpit's features:"
    )

    st.markdown("""
        - **EDA & Insights:** Start here to explore the historical data and understand the key patterns driving customer churn across the bank.
        - **Predictions & Actions:** Upload a list of customers to get real-time churn scores and actionable recommendations.
        - **ROI Simulation:** Go to the 'Predictions' tab to fine-tune your retention strategy based on campaign costs and customer value.
    """)

    st.markdown("---")

    with st.expander("Model Card", expanded=False):
        st.subheader("Model Card")
        md = (APP_DIR / "model_card.md")
        if md.exists():
            st.markdown(md.read_text())
        else:
            st.info("model_card.md not found. Create one describing data, preprocessing, model, metrics, intended use, limitations, fairness, and monitoring.")



def page_eda():
    st.header("Manager's EDA & Churn Insights")
    st.caption("An automated analysis of customer data to identify the key drivers of churn.")

    up = st.file_uploader("Upload BankChurners.csv", type=["csv"], key="eda_csv")
    if not up:
        st.info("Please upload the Bank Customers(csv) dataset to begin the analysis.")
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

    # --------------------------------------
    # 1. EXECUTIVE KPI DASHBOARD
    # --------------------------------------
    st.markdown("### Overview")
    c1, c2, c3 = st.columns(3)
    churn_rate = df["Churn"].mean()
    with c1:
        st.metric("Total Customers", f"{len(df):,}")
    with c2:
        st.metric("Overall Churn Rate", f"{churn_rate:.1%}")
    with c3:
        st.metric("Retained Customers", f"{len(df[df['Churn']==0]):,}")

    st.markdown("#### Customer Distribution by Segment")
    bal = df["Churn"].value_counts().rename(index={0:"Existing",1:"Attrited"}).reset_index()
    bal.columns = ["Class","Count"]
    fig_bal = px.bar(bal, x="Class", y="Count", text="Count", title="Proportion of Existing vs. Attrited (Churned) Customers")
    fig_bal.update_traces(textposition="outside")
    st.plotly_chart(fig_bal, use_container_width=True)

    st.markdown("---")

    # ----------------------
    # Distributions (by churn)
    # ----------------------
    st.markdown("### Distributions by churn")
    default_nums = [c for c in ["Total_Trans_Ct","Total_Trans_Amt","Total_Ct_Chng_Q4_Q1","Total_Amt_Chng_Q4_Q1","Months_Inactive_12_mon"] if c in num_cols][:2]
    sel_num = st.multiselect("Numeric features for distribution", options=num_cols, default=default_nums, key="eda_dist_nums")
    if sel_num:
        for col in sel_num:
            fig = px.histogram(df, x=col, color=df["Churn"].map({0:"Existing",1:"Attrited"}),
                               barmode="overlay", nbins=50,
                               title=f"Distribution of {col} by Churn")
            fig.update_layout(legend_title_text="Class")
            fig.update_traces(opacity=0.65)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"This chart shows how the values of **{sel_num}** differ between customers "
                       f"who stayed and those who left. Look for clear separations in the distributions.")
            
            st.markdown("---")


    # ----------------------
    # Auto-Insights (quick highlights)
    # ----------------------
    with st.expander("Auto-Insights (top signals detected)", expanded=False):
        st.info(
            "This automated report analyzes all customer attributes to find the strongest signals "
            "that predict whether a customer will leave."
        )
        
        bullets = []

        # --- Analysis for Numeric Drivers ---
        mean_diffs = []
        for col in num_cols:
            if df[col].nunique() > 1: # Ensure there is variance to compare
                grouped = df.groupby("Churn")[col].mean()
                if 0 in grouped.index and 1 in grouped.index:
                    # Calculate the difference between churned and existing customers' means
                    mean_diffs.append((col, float(grouped[1]), float(grouped[0])))
        
        # Normalize the differences by the standard deviation to find the most impactful changes
        mean_diffs_sorted = sorted(mean_diffs, 
                                key=lambda x: abs((x[1] - x[2]) / df[x[0]].std()), 
                                reverse=True)

        top_numeric_drivers = mean_diffs_sorted[:4]
        if top_numeric_drivers:
            numeric_insights = []
            for col, churn_mean, existing_mean in top_numeric_drivers:
                diff = churn_mean - existing_mean
                direction = "lower" if diff < 0 else "higher"
                # Format the insight into a clear, readable sentence
                insight = (f"**{col.replace('_', ' ')}**: On average, churned customers had a "
                        f"<u>significantly {direction}</u> value (${abs(diff):,.0f}) "
                        f" than existing customers (${existing_mean:,.0f}).")
                numeric_insights.append(insight)
            bullets.append(("Top Predictive Behaviors (Numeric)", numeric_insights))

        # --- Analysis for Categorical Drivers ---
        highest_churn_segments = []
        for col in cat_cols:
            if df[col].nunique() > 1:
                churn_rates = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
                if not churn_rates.empty:
                    # Identify the segment with the highest and lowest churn rates
                    highest_segment = churn_rates.index[0]
                    highest_rate = churn_rates.iloc[0]
                    lowest_segment = churn_rates.index[-1]
                    lowest_rate = churn_rates.iloc[-1]
                    avg_rate = df["Churn"].mean()
                    # Add insight only if the churn rate is meaningfully different from the average
                    if highest_rate > avg_rate * 1.2:
                        highest_churn_segments.append((col, highest_segment, highest_rate, lowest_segment, lowest_rate))

        top_cat_drivers = sorted(highest_churn_segments, key=lambda x: x[2], reverse=True)[:4]
        if top_cat_drivers:
            cat_insights = []
            for col, segment, rate, low_seg, low_rate in top_cat_drivers:
                insight = (f"**{col.replace('_', ' ')}**: The customer segment with the highest churn risk is "
                        f"<u>'{segment}'</u>, with a churn rate of **{rate:.1%}**. "
                        f"(For comparison, the lowest risk segment is '{low_seg}' at {low_rate:.1%})")
                cat_insights.append(insight)
            bullets.append(("Highest Risk Customer Segments (Categorical)", cat_insights))

        # --- Display the Automated Insights ---
        for title, insights_list in bullets:
            st.subheader(title)
            for insight in insights_list:
                st.markdown(f"- {insight}", unsafe_allow_html=True)
            st.write("") # Add some space

    st.markdown("---")


    # ----------------------
    # Churn rate vs feature (numeric via quantile bins)
    # ----------------------
    st.markdown("### Churn rate vs feature")
    c1, c2 = st.columns(2)
    with c1:
        sel_num_cr = st.selectbox("Select a numeric feature to analyze:", options=num_cols, index=(num_cols.index("Total_Trans_Ct") if "Total_Trans_Ct" in num_cols else 0) if num_cols else 0, key="eda_churn_num")
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
            with st.expander("💡**How to read this chart**", expanded=False):
                st.caption(f"Customers are grouped into five bins based on their "
                   f"'{sel_num_cr}' value, from lowest to highest. The line shows the actual churn rate for each group. "
                   f"A clear upward or downward trend is a strong indicator of how this factor influences churn.")
    with c2:
        # Categorical churn rate
        cat_opts = [c for c in cat_cols if c not in ["Attrition_Flag"]]
        sel_cat_cr = st.selectbox("Select a categorical feature to analyze:", options=cat_opts, index=(cat_opts.index("Card_Category") if "Card_Category" in cat_opts else 0) if cat_opts else 0, key="eda_churn_cat")
        if cat_opts:
            tmp = df.groupby(sel_cat_cr)["Churn"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(tmp, x=sel_cat_cr, y="Churn", title=f"Churn rate by {sel_cat_cr}")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("💡**How to read this chart**", expanded=False):
                st.caption(f"This bar chart compares the churn rate across different customer groups based on '{sel_cat_cr}'. "
                    f"A taller bar means a higher percentage of customers in that specific group have left the bank, highlighting it as a **higher-risk segment** that may require targeted retention efforts.")

    st.markdown("---")

    # ----------------------
    # Outlier diagrams (box/violin)
    # ----------------------
    st.markdown("### Outliers (box / violin)")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        col_box = st.selectbox("Numeric feature (boxplot)", options=num_cols, index=(num_cols.index(default_nums[0]) if default_nums else 0) if num_cols else 0, key="eda_box")
        if num_cols:
            fig = px.box(df, x=df["Churn"].map({0:"Existing",1:"Attrited"}), y=col_box,
                            title=f"Boxplot of {col_box} by Churn")
            fig.update_layout(xaxis_title="Class")
            st.plotly_chart(fig, use_container_width=True)
    with col_out2:
        col_violin = st.selectbox("Numeric feature (violin)", options=num_cols, index=(num_cols.index(default_nums[1]) if len(default_nums)>1 else 0) if num_cols else 0, key="eda_violin")
        if num_cols:
            fig = px.violin(df, x=df["Churn"].map({0:"Existing",1:"Attrited"}), y=col_violin, box=True, points="outliers",
                            title=f"Violin plot of {col_violin} by Churn")
            fig.update_layout(xaxis_title="Class")
            st.plotly_chart(fig, use_container_width=True)



    # ----------------------
    # Correlation matrix (numeric features)
    # ----------------------
    # with st.expander("See How Numeric Features Relate to Each Other"):
    #     st.markdown("### Correlation matrix (numeric features)")
    #     num_for_corr = ["Churn"] + num_cols
    #     corr = df[num_for_corr].corr(numeric_only=True)
    #     fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower",
    #                     title="Correlation heatmap (including Churn)")
    #     st.plotly_chart(fig, use_container_width=True)

    # --- It's good practice to define your labels in one place ---

FEATURE_LABELS = {
    "Total_Trans_Ct": "Total Transaction Count (Last 12 Months)",
    "Total_Ct_Chng_Q4_Q1": "Change in Transaction Count (Q4 vs Q1)",
    "Total_Amt_Chng_Q4_Q1": "Change in Transaction Amount (Q4 vs Q1)",
    "Months_Inactive_12_mon": "Months Inactive (Last 12 Months)",
    "Contacts_Count_12_mon": "Contacts with Bank (Last 12 Months)",
    "Total_Relationship_Count": "Total Number of Products with Bank",
    "Credit_Limit": "Credit Card Limit ($)",
    "Total_Revolving_Bal": "Total Revolving Balance ($)",
    "Income_Category": "Annual Income Category",
    "Card_Category": "Credit Card Type"
}

def page_predictor(pipe, expected_cols):
    st.header("Predictor")
    tab1, tab2 = st.tabs(["📦 Batch Scoring (recommended)", "👤 Single Customer (what‑if)"])

    with tab1:
        # --- ROI controls ---
        st.info("Set the financial assumptions for the ROI-driven analysis.")
        c1, c2 = st.columns([1, 1])
        with c1:
            value = st.number_input("Average Value per Retained Customer ($)", 
                                    min_value=0.0, value=200.0, step=10.0, key="bs_value",
                                    help="Estimate the average profit or lifetime value gained from successfully retaining a single customer.")
        with c2:
            cost = st.number_input("Cost per Retention Contact ($)", 
                                min_value=0.0, value=2.0, step=1.0, key="bs_cost",
                                help="The average cost of a single retention action (e.g., a phone call, a special offer).")

        eff = st.slider("Retention Campaign Effectiveness (α)", 0.0, 1.0, 0.30, 0.05, key="bs_eff",
                        help="What percentage of customers who are contacted for retention will actually stay? (e.g., 30%)")

        st.markdown("---")

        st.write("Upload a CSV of current customers. The model will score them and create a prioritized retention list.")
        up = st.file_uploader("Upload Customer List (CSV)", type=["csv"], key="batch")
        if up:
            df_in = pd.read_csv(up)
            df_in = apply_derived_fields(df_in)
            proba = score(df_in, pipe, expected_cols)

            t_star, ev_star, ts, evs = recommend_threshold(proba, value, cost, eff)
            st.success(f"Recommended Threshold: **{t_star:.2f}** | Max Expected Value from Campaign: **${ev_star:,.0f}**")
            
            threshold = st.slider("Decision Threshold", 0.1, 0.9, float(t_star), 0.01, key="thr",
                                help="Set the churn probability threshold. Any customer above this will be flagged for retention.")
            
            label_numeric = (proba >= threshold).astype(int)
            label_text = np.where(label_numeric == 1, "Attrited", "Existing")
            
            enriched = df_in.copy()
            enriched["churn_probability"] = proba
            enriched["predicted_label"] = label_text
            
            reasons_list, actions = [], []
            for i in range(len(enriched)):
                info = reason_codes(enriched.iloc[i], proba[i])
                reasons_list.append(" • ".join(info["reasons"]))
                actions.append(info["action"])
                
            enriched["top_reasons"] = reasons_list
            enriched["recommended_action"] = actions
            
            id_cols = [c for c in enriched.columns if c.lower() in ("clientnum", "customer_id", "id")]
            show_cols = id_cols + ["churn_probability", "predicted_label", "top_reasons", "recommended_action"]
            
            st.markdown("#### Ranked Retention Queue")

            # --- THIS IS THE KEY CHANGE ---
            # The dataframe is now sorted but NOT filtered, so it shows all customers.
            retention_full_list = enriched.sort_values("churn_probability", ascending=False)
            
            st.dataframe(retention_full_list[show_cols].head(50), use_container_width=True)
            st.download_button("Download Full Results CSV", data=enriched.to_csv(index=False).encode("utf-8"),
                            file_name="retention_queue.csv", mime="text/csv")
        else:
            st.info("Awaiting customer list upload…")

    with tab2:
        st.caption("Enter the customer's details to get a real-time churn prediction.")
        c1, c2 = st.columns(2)
        
        with c1:
            total_trans_ct = st.number_input(
                label=FEATURE_LABELS["Total_Trans_Ct"], 
                min_value=0, value=12, step=1,
                help="The total number of transactions the customer made in the last year."
            )
            total_ct_chg_q4_q1 = st.number_input(
                label=FEATURE_LABELS["Total_Ct_Chng_Q4_Q1"], 
                min_value=0.0, max_value=5.0, value=0.45, step=0.05,
                help="The percentage change in the number of transactions from the first quarter to the fourth quarter."
            )
            total_amt_chg_q4_q1 = st.number_input(
                label=FEATURE_LABELS["Total_Amt_Chng_Q4_Q1"], 
                min_value=0.0, max_value=5.0, value=0.50, step=0.05,
                help="The percentage change in the total dollar amount of transactions from the first quarter to the fourth quarter."
            )
            months_inactive = st.number_input(
                label=FEATURE_LABELS["Months_Inactive_12_mon"], 
                min_value=0, max_value=12, value=5, step=1,
                help="How many of the last 12 months the customer has been inactive."
            )
            contacts_12_mon = st.number_input(
                label=FEATURE_LABELS["Contacts_Count_12_mon"], 
                min_value=0, max_value=20, value=5, step=1,
                help="How many times the customer has contacted the bank in the last year."
            )

        with c2:
            rel_count = st.number_input(
                label=FEATURE_LABELS["Total_Relationship_Count"], 
                min_value=0, max_value=10, value=1, step=1,
                help="The total number of products the customer has with the bank (e.g., credit card, savings account)."
            )
            credit_lim = st.number_input(
                label=FEATURE_LABELS["Credit_Limit"], 
                min_value=200.0, max_value=100000.0, value=2000.0, step=100.0
            )
            revol_bal = st.number_input(
                label=FEATURE_LABELS["Total_Revolving_Bal"], 
                min_value=0.0, max_value=100000.0, value=500.0, step=50.0,
                help="The portion of the credit limit that is currently being used."
            )
            income = st.selectbox(
                label=FEATURE_LABELS["Income_Category"], 
                options=CAT_OPTIONS["Income_Category"], index=1
            )
            card = st.selectbox(
                label=FEATURE_LABELS["Card_Category"], 
                options=CAT_OPTIONS["Card_Category"], index=0
            )
        
        # --- The backend logic remains the same ---
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
        info = reason_codes(df1.iloc[0], proba)

        st.metric("Churn probability", f"{proba:.1%}")
        st.write("**Top reasons for this prediction**:", " | ".join(info["reasons"]))
        st.write("**Recommended action**:", info["action"])

        st.write("---")
        st.markdown("#### What-if Analysis: The Power of Engagement")
        
        # Use the human-readable label for the slider as well
        new_ct = st.slider(
            f"Simulate increasing the '{FEATURE_LABELS['Total_Trans_Ct']}'", 
            0, 100, int(total_trans_ct), 1,
            help="Use this slider to see how the churn probability changes if you can encourage this customer to make more transactions."
        )
        
        df2 = df1.copy()
        df2.loc[0, "Total_Trans_Ct"] = new_ct
        proba2 = score(df2, pipe, expected_cols)[0]
        delta = proba2 - proba
        
        st.metric("New Churn Probability", f"{proba2:.1%}", delta=f"{delta:+.1%}", 
                help="The 'delta' shows the change from the original prediction. A negative delta is an improvement.")

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
    md = (APP_DIR / "model_card.md")
    if md.exists():
        st.markdown(md.read_text())
    else:
        st.info("model_card.md not found. Create one describing data, preprocessing, model, metrics, intended use, limitations, fairness, and monitoring.")

def main():
    # value, cost, eff, tab = sidebar_controls()
    # tab = sidebar_controls()
    tab = sidebar_nav()
    st.markdown("<div class='topbar'>Bank Churn Prediction</div>", unsafe_allow_html=True)

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
        # page_predictor(pipe, expected_cols, value, cost, eff)
        page_predictor(pipe, expected_cols)

if __name__ == "__main__":
    main()
