import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def train_and_export_model():
    """
    Loads the BankChurners dataset, trains the XGBoost model pipeline,
    and exports the trained model and feature importance plot.
    This script is based on the logic from the provided FDM.ipynb notebook.
    """
    # 1. Load and Prepare Data
    df = pd.read_csv("BankChurners.csv")
    df = df.drop(columns=['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])
    df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

    X = df.drop('Attrition_Flag', axis=1)
    y = df['Attrition_Flag']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # 2. Define Preprocessing Pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. Define the Model Pipeline with SMOTE and XGBoost
    # Using best parameters found in the notebook
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(**xgb_params, random_state=42))
    ])

    # 4. Train the Model on the entire dataset
    print("Training the model on the full dataset...")
    model_pipeline.fit(X, y)
    print("Model training complete.")

    # 5. Export the trained pipeline
    model_filename = 'churn_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"Model pipeline saved as '{model_filename}'")

    # 6. Generate and Save Feature Importance Plot
    try:
        # Get feature names after one-hot encoding
        ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = list(numerical_features) + list(ohe_feature_names)
        
        importances = model_pipeline.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        
        plot_filename = 'feature_importance.png'
        plt.savefig(plot_filename)
        print(f"Feature importance plot saved as '{plot_filename}'")
    except Exception as e:
        print(f"Could not generate feature importance plot. Error: {e}")


if __name__ == "__main__":
    # To run this script:
    # 1. Make sure 'BankChurners.csv' is in the same directory.
    # 2. Run 'pip install pandas joblib scikit-learn xgboost imbalanced-learn matplotlib seaborn'
    # 3. Execute 'python train_model.py' in your terminal.
    # This will create 'churn_model.joblib' and 'feature_importance.png'.
    train_and_export_model()
