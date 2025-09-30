import streamlit as st
import pandas as pd
import joblib
import os

# --- Model & Data Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads the trained model pipeline."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run `train_model.py` first.")
        return None
    return joblib.load(model_path)

# --- API Call to Gemini (Placeholder) ---
# In a real scenario, this would involve API keys and proper error handling.
# For this example, we simulate the call for demonstration purposes.
def get_gemini_recommendations(data):
    """
    Generates a personalized retention plan using a simulated Gemini API call.
    """
    # This is a simplified representation. A real implementation would call the Gemini API.
    # The prompt engineering would be more sophisticated.
    
    # Find top 2-3 risk factors based on simple rules
    risks = []
    if data['Total_Trans_Ct'] < 60:
        risks.append(f"low transaction count ({data['Total_Trans_Ct']})")
    if data['Months_Inactive_12_mon'] >= 3:
        risks.append(f"high inactivity ({data['Months_Inactive_12_mon']} months)")
    if data['Contacts_Count_12_mon'] >= 3:
        risks.append(f"multiple support contacts ({data['Contacts_Count_12_mon']})")
    if data['Total_Relationship_Count'] <= 2:
        risks.append(f"few products held ({data['Total_Relationship_Count']})")

    risk_str = ", ".join(risks) if risks else "general disengagement"

    # Simulated response from Gemini based on the identified risks
    plan = f"""
    ### **Personalized AI Retention Strategy**

    Based on the key risk factors of **{risk_str}**, here is a suggested action plan:

    - **🎁 Re-engagement Offer:** Proactively reach out with a targeted spending offer, such as **10% cashback** on their most frequent spending category for the next month, to encourage transaction activity.

    - **📞 Proactive Wellness Call:** Schedule a call from a senior customer service agent to address any lingering issues from their previous contacts and to reinforce the value of their relationship with the bank.

    - **💡 Product Awareness:** Introduce them to a complementary product that fits their profile, like a high-yield savings account or investment opportunity, to increase their total relationship count and embeddedness with the bank.
    """
    return plan

# --- UI for Prediction Page ---
def predictor_page():
    st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="wide")
    st.title("🔮 Churn Predictor")
    
    model = load_model('churn_model.joblib')
    if model is None:
        st.stop()

    # Define unique values for categorical dropdowns
    cat_options = {
        'Gender': ['M', 'F'],
        'Education_Level': ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', 'Unknown'],
        'Marital_Status': ['Married', 'Single', 'Divorced', 'Unknown'],
        'Income_Category': ['$60K - $80K', 'Less than $40K', '$80K - $120K', '$40K - $60K', '$120K +', 'Unknown'],
        'Card_Category': ['Blue', 'Silver', 'Gold', 'Platinum']
    }

    # Sidebar for prediction mode
    mode = st.sidebar.radio("Choose Prediction Mode", ("Single Customer", "Batch Upload"))
    threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)

    if mode == "Single Customer":
        st.header("Enter Customer Details for a Real-Time Prediction")
        
        with st.form("prediction_form"):
            input_data = {}
            
            # Use columns for a cleaner layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Personal Info")
                input_data['Customer_Age'] = st.slider("Age", 18, 100, 45)
                input_data['Gender'] = st.selectbox("Gender", options=cat_options['Gender'])
                input_data['Dependent_count'] = st.slider("Dependents", 0, 10, 2)
                input_data['Education_Level'] = st.selectbox("Education Level", options=cat_options['Education_Level'])
                input_data['Marital_Status'] = st.selectbox("Marital Status", options=cat_options['Marital_Status'])
                input_data['Income_Category'] = st.selectbox("Income Category", options=cat_options['Income_Category'])

            with col2:
                st.subheader("Bank Relationship")
                input_data['Card_Category'] = st.selectbox("Card Category", options=cat_options['Card_Category'])
                input_data['Months_on_book'] = st.number_input("Months as Customer", value=36, min_value=0)
                input_data['Total_Relationship_Count'] = st.slider("Total Products Held", 1, 10, 4)
                input_data['Months_Inactive_12_mon'] = st.slider("Months Inactive (Last 12m)", 0, 12, 2)
                input_data['Contacts_Count_12_mon'] = st.slider("Contacts (Last 12m)", 0, 10, 2)

            with col3:
                st.subheader("Financials & Usage")
                input_data['Credit_Limit'] = st.number_input("Credit Limit ($)", value=10000.0, step=500.0)
                input_data['Total_Revolving_Bal'] = st.number_input("Total Revolving Balance ($)", value=1500.0)
                # --- FIXED: Added the two missing input fields below ---
                input_data['Avg_Open_To_Buy'] = st.number_input("Average Open To Buy ($)", value=8500.0, help="Credit Limit - Total Revolving Balance")
                input_data['Total_Amt_Chng_Q4_Q1'] = st.number_input("Transaction Amount Change (Q4 vs Q1)", value=0.8, help="Change in transaction amount from Q1 to Q4")
                # --- End of fix ---
                input_data['Total_Trans_Amt'] = st.number_input("Total Transaction Amount (Last 12m)", value=4000.0)
                input_data['Total_Trans_Ct'] = st.number_input("Total Transaction Count (Last 12m)", value=80)
                input_data['Total_Ct_Chng_Q4_Q1'] = st.number_input("Transaction Count Change (Q4 vs Q1)", value=0.7)
                input_data['Avg_Utilization_Ratio'] = st.slider("Avg. Card Utilization Ratio", 0.0, 1.0, 0.15)
                
            submitted = st.form_submit_button("Predict Churn")

        if submitted:
            # Prepare data for prediction
            input_df = pd.DataFrame([input_data])
            
            # Predict
            prob = model.predict_proba(input_df)[0][1]
            prediction = 1 if prob >= threshold else 0

            # Display results
            st.markdown("---")
            st.header("Prediction Result")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if prediction == 1:
                    st.error("Prediction: LIKELY TO CHURN", icon="🚨")
                else:
                    st.success("Prediction: UNLIKELY TO CHURN", icon="✅")
            
            with res_col2:
                st.metric(label="Churn Probability", value=f"{prob:.2%}")

            if prediction == 1:
                st.warning("This customer is at high risk of churning. Consider the following AI-driven retention plan.")
                with st.spinner("✨ Generating AI Retention Plan with Gemini..."):
                    retention_plan = get_gemini_recommendations(input_data)
                    st.markdown(retention_plan)

    elif mode == "Batch Upload":
        st.header("Upload a CSV File for Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(df_batch.head())

            if st.button("Run Batch Prediction"):
                # Make sure the uploaded file has the necessary columns before predicting
                expected_cols = model.named_steps['preprocessor'].feature_names_in_
                if not set(expected_cols).issubset(df_batch.columns):
                    missing_cols = set(expected_cols) - set(df_batch.columns)
                    st.error(f"The uploaded CSV is missing the following required columns: {missing_cols}")
                else:
                    with st.spinner("Processing batch..."):
                        # Predict probabilities for the entire file
                        probs = model.predict_proba(df_batch)[:, 1]
                        df_batch['churn_probability'] = probs
                        df_batch['predicted_churn'] = (probs >= threshold).astype(int)
                        
                        st.success("Batch prediction complete!")
                        st.dataframe(df_batch.head())
                        
                        # Provide download link
                        csv = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name='churn_predictions.csv',
                            mime='text/csv',
                        )

if __name__ == "__main__":
    predictor_page()

