import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# Create requirements.txt
def create_requirements():
    """Create a requirements.txt file dynamically."""
    requirements = """streamlit
pandas
numpy
scikit-learn
shap
matplotlib
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements)

# Load the dataset
def load_data():
    """Load and preprocess the hate crime dataset."""
    data = pd.read_csv('hate_crime.csv')

    # Handle missing values
    data['adult_victim_count'] = data['adult_victim_count'].fillna(0)
    data['juvenile_victim_count'] = data['juvenile_victim_count'].fillna(0)

    # Convert dates to datetime
    data['incident_date'] = pd.to_datetime(data['incident_date'], errors='coerce')

    # Encode categorical variables
    categorical_cols = ['bias_desc', 'offender_race', 'offender_ethnicity', 'location_name']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    return data

# Data splitting
def prepare_data(data):
    """Prepare data for training and testing."""
    feature_cols = [col for col in data.columns if col not in ['incident_id', 'data_year', 'state_name', 'incident_date', 'victim_types']]
    X = data[feature_cols]
    y = (data['victim_count'] > 1).astype(int)  # Example target: more than 1 victim
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Build Streamlit app
def main():
    st.title("Hate Crime Hotspot Predictor")

    # Create requirements.txt
    create_requirements()

    # Load data
    data = load_data()
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    # Input filters
    st.sidebar.title("Filters")
    selected_state = st.sidebar.selectbox("Select a State", options=data['state_name'].unique())
    
    # Display filtered data
    filtered_data = data[data['state_name'] == selected_state]
    st.write(f"### Data for {selected_state}")
    st.dataframe(filtered_data.head())

    # Train Logistic Regression Model
    st.write("### Logistic Regression Model")
    X_train, X_test, y_train, y_test = prepare_data(data)
    logreg = LogisticRegression(max_iter=500)
    logreg.fit(X_train, y_train)
    st.write("Model trained on demographic and bias data.")

    # Feature Importance for Logistic Regression
    st.write("#### Logistic Regression Coefficients")
    coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": logreg.coef_[0]}).sort_values(by="Coefficient", ascending=False)
    st.bar_chart(coef_df.set_index("Feature"))

    # SHAP Explainability
    st.write("### SHAP Analysis")
    explainer = shap.Explainer(logreg, X_train)
    shap_values = explainer(X_test)
    st.write("#### SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot()

if __name__ == "__main__":
    main()
