import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    """Loads the dataset and performs initial cleaning."""
    df = pd.read_csv("BankChurners.csv")
    df = df.drop(columns=['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])
    df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda x: "Churned" if x == 'Attrited Customer' else "Existing")
    return df

def eda_page():
    st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊", layout="wide")
    st.title("📊 Exploratory Data Analysis")
    st.write("This section provides a deep dive into the dataset, exploring the distributions of key features and their relationship with customer churn.")

    df = load_data()

    # --- Section 1: Dataset Overview ---
    st.header("1. Dataset Overview")
    st.dataframe(df.head())

    with st.expander("Click here to see feature descriptions"):
        st.markdown("""
        - **Attrition_Flag**: Whether the customer has churned (Attrited Customer) or not (Existing Customer).
        - **Customer_Age**: The age of the customer in years.
        - **Gender**: 'M' for Male, 'F' for Female.
        - **Dependent_count**: Number of dependents.
        - **Education_Level**: Educational qualification of the customer.
        - **Marital_Status**: Marital status of the customer.
        - **Income_Category**: Annual income category of the customer.
        - **Card_Category**: Type of card held by the customer (e.g., Blue, Silver, Gold).
        - **Months_on_book**: Period of relationship with the bank (in months).
        - **Total_Relationship_Count**: Total number of products held by the customer.
        - **Months_Inactive_12_mon**: Number of months inactive in the last 12 months.
        - **Contacts_Count_12_mon**: Number of contacts in the last 12 months.
        - **Credit_Limit**: The credit limit on the credit card.
        - **Total_Revolving_Bal**: Total revolving balance on the credit card.
        - **Total_Trans_Amt**: Total transaction amount in the last 12 months.
        - **Total_Trans_Ct**: Total transaction count in the last 12 months.
        - **Avg_Utilization_Ratio**: Average card utilization ratio.
        """)

    # --- Section 2: Churn Distribution ---
    st.header("2. Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Attrition_Flag', data=df, palette='pastel', ax=ax)
    ax.set_title('Overall Customer Churn Distribution')
    st.pyplot(fig)
    st.markdown("**Insight:** The dataset is imbalanced, with a significantly larger number of existing customers than churned customers. This is a crucial observation that was addressed during model training using SMOTE.")

    st.header("3. Exploratory Data Analysis (EDA)")
    
    # --- Section 3: Demographic Analysis ---
    st.subheader("3.1 Demographic & Categorical Feature Analysis")
    
    cat_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    
    # Create 2 columns for a better layout
    col1, col2 = st.columns(2)
    columns = [col1, col2]

    for i, feature in enumerate(cat_features):
        with columns[i % 2]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(y=feature, hue='Attrition_Flag', data=df, palette='viridis', ax=ax)
            ax.set_title(f'Churn Distribution by {feature}')
            plt.tight_layout()
            st.pyplot(fig)
    st.markdown("**Insight:** While churn occurs across all demographic groups, we can observe slight variations. For instance, customers with 'Doctorate' or 'Post-Graduate' education levels appear less likely to churn compared to others.")


    # --- Section 4: Numerical Feature Analysis ---
    st.header("4. Numerical Feature Analysis")
    st.write("Comparing the distributions of key numerical features for churned vs. existing customers.")
    
    num_features = ['Total_Trans_Ct', 'Total_Trans_Amt', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Total_Revolving_Bal']

    for i, feature in enumerate(num_features):
        with columns[i % 2]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=df, x=feature, hue='Attrition_Flag', multiple='stack', kde=True, palette='magma')
            ax.set_title(f'Distribution of {feature}')
            plt.tight_layout()
            st.pyplot(fig)
            
    st.markdown("""
    **Key Insights from Numerical Features:**
    - **Transaction Count & Amount:** Customers who churned (`Attrition_Flag=1`) have a distribution centered around lower transaction counts and amounts. This is a very strong indicator of churn.
    - **Inactivity & Contacts:** Churned customers tend to have been inactive for more months and have had more contacts with the bank in the last year, suggesting dissatisfaction or issues.
    - **Revolving Balance:** Customers who churn often have a lower revolving balance, indicating they are using the card less.
    """)
    
    # --- Section 5: Correlation Matrix ---
    st.header("5. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 10))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix of Numerical Features')
    st.pyplot(fig)
    st.markdown("**Insight:** Features like `Total_Trans_Amt` and `Total_Trans_Ct` are highly correlated, which is expected. The heatmap helps to identify multicollinearity, which tree-based models like XGBoost are generally robust against.")


if __name__ == "__main__":
    eda_page()
