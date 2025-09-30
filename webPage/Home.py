import streamlit as st
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def home_page():
    st.set_page_config(
        page_title="Customer Churn Predictor",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                color: #2c3e50;
                font-weight: bold;
                text-align: center;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #34495e;
                text-align: center;
                margin-bottom: 2rem;
            }
            .stButton>button {
                background-color: #3498db;
                color: white;
                border-radius: 12px;
                padding: 10px 20px;
                border: none;
                font-size: 16px;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                background-color: #2980b9;
                transform: scale(1.05);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">💳 Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An Interactive Tool for Proactive Customer Retention</p>', unsafe_allow_html=True)

    st.info("""
        **Welcome to the Customer Churn Prediction Dashboard!**

        This application leverages a powerful XGBoost machine learning model to predict the likelihood of a credit card customer churning (closing their account). By identifying at-risk customers, businesses can implement targeted retention strategies to improve customer loyalty and reduce financial loss.

        **Navigate through the app using the sidebar to:**
        - **Explore the Data:** Understand the dataset and key factors influencing churn through interactive visualizations.
        - **Predict Churn:** Get real-time predictions for individual customers or upload a batch of customer data.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Project Objective")
        st.write("""
            The primary goal of this project is to build a reliable classification model that accurately predicts customer churn. This tool serves as the software solution deliverable, providing a user-friendly interface for bank stakeholders to make data-driven decisions.
        """)

    with col2:
        st.subheader("🚀 Technology Stack")
        st.write("""
            - **Model:** Tuned `XGBoost` Classifier with `SMOTE` for handling class imbalance.
            - **App Framework:** `Streamlit` for creating this interactive web application.
            - **Data Manipulation:** `Pandas` and `Scikit-learn`.
            - **AI Strategy:** `Google Gemini` for generating personalized retention plans.
        """)
        
    st.success("Please use the navigation panel on the left to explore the different sections of the application.")


if __name__ == "__main__":
    home_page()
