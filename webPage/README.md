Professional Churn Prediction Application
This project contains a multi-page Streamlit application for predicting customer churn using a pre-trained XGBoost model. It includes data exploration, single-customer prediction, batch processing, and an AI-powered retention strategy generator using the Gemini API.

🚀 How to Run This Application
1. Initial Setup
First, place all the provided files in a single project folder. Your folder structure should look like this:

/your-project-folder/
|-- Home.py                     # Main streamlit app file
|-- train_model.py              # Script to train and export the model
|-- BankChurners.csv            # The dataset
|-- README.md                   # This instruction file
|-- pages/
|   |-- 1_Exploratory_Data_Analysis.py
|   `-- 2_Churn_Predictor.py

2. Install Dependencies
Open your terminal or command prompt, navigate to your-project-folder, and install the required Python libraries:

pip install streamlit pandas joblib scikit-learn xgboost imbalanced-learn matplotlib seaborn

3. Train the Model and Generate Artifacts
Before you can run the app, you need to train the model. This step will create two essential files: churn_model.joblib and feature_importance.png.

Run the following command in your terminal:

python train_model.py

After this command finishes, you will see the two new files in your project folder.

4. Run the Streamlit Application
You are now ready to launch the interactive application. Run this command in your terminal:

streamlit run Home.py

Your web browser will automatically open with the application running. You can now explore the different pages and use the churn predictor.

File Descriptions
train_model.py: A script that encapsulates the entire model training process from your notebook. It loads data, preprocesses it, trains the XGBoost model with SMOTE, and saves the final model pipeline (churn_model.joblib) and a feature importance plot (feature_importance.png).

Home.py: The landing page for the Streamlit application. It provides an introduction to the project.

pages/1_Exploratory_Data_Analysis.py: The EDA page. It loads the dataset and displays various plots and insights, recreating the analysis from your notebook in an interactive format.

pages/2_Churn_Predictor.py: The core prediction tool. It allows for both single and batch predictions and integrates with the (simulated) Gemini API to provide retention strategies.

BankChurners.csv: The raw dataset used for training and analysis.