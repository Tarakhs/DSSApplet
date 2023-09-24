# Panagiotarakis Nikos P2019152
# DSS 2023

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

clf = None

# Data Preprocessing Check
data_preprocessed = False

def load_data(file):
    data = pd.read_csv(file)
    return data

def preprocess_data(data):
    label_encoder = LabelEncoder()
    data["Classification"] = label_encoder.fit_transform(data["Classification"]) + 1
    return data

def get_cell_style(classification, predicted_classification):
    if predicted_classification == 1:
        return "background-color: rgba(0, 255, 0, 0.3);"
    elif predicted_classification == 2:
        return "background-color: rgba(255, 0, 0, 0.3);"
    elif classification == 1:
        return "background-color: rgba(0, 128, 0, 0.3);"
    elif classification == 2:
        return "background-color: rgba(255, 0, 0, 0.3);"
    else:
        return ""

def main():
    global clf
    global data_preprocessed

    st.title("Breast Cancer Detection App")

    # File Uploading
    file = st.file_uploader("Upload CSV File:", type=["csv"])

    if file is not None:
        data = load_data(file)

        st.write("## Uploaded Data")
        st.write(data)

        st.sidebar.title("Data Processing Options")
        preprocess_data_checkbox = st.sidebar.checkbox("Pre-Process Data")

        if preprocess_data_checkbox:
            data = preprocess_data(data)
            data_preprocessed = True  # Variable Updating
            st.write("Data has been preprocessed ✅")

        st.sidebar.title("Model Selection")
        algorithm = st.sidebar.selectbox("Select Classification Algorithm", ["Random Forest", "Support Vector Machine", "K-Nearest Neighbors"])

        if st.button("Train Model"):
            if algorithm == "Random Forest":
                # Data Preparation
                X = data.drop(columns=["Classification"])
                y = data["Classification"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Classifier Training
                clf = RandomForestClassifier(random_state=42)
                clf.fit(X_train, y_train)

                # Calculate Algorithm Accuracy
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.write(f"Random Forest Classifier Accuracy: {accuracy:.2f}")

                # Output Trained Model
                joblib.dump(clf, 'trained_model.joblib')

            elif algorithm == "K-Nearest Neighbors":
                
                X = data.drop(columns=["Classification"])
                y = data["Classification"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                clf = KNeighborsClassifier()
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.write(f"K-Nearest Neighbors Classifier Accuracy: {accuracy:.2f}")

                joblib.dump(clf, 'trained_model.joblib')

            elif algorithm == "Support Vector Machine":

                X = data.drop(columns=["Classification"])
                y = data["Classification"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                clf = SVC()
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.write(f"Support Vector Machine Classifier Accuracy: {accuracy:.2f}")

                joblib.dump(clf, 'trained_model.joblib')

        # Results Table
        if clf is not None:
            st.write("## Classification Results")
            st.write("The table below shows the results for the uploaded data.")

            y_all_pred = clf.predict(X)
            data["Predicted Classification"] = y_all_pred

            data_display = data[["Classification", "Predicted Classification"]].copy()

            data_display = data_display.style.applymap(lambda x: get_cell_style(x, x), subset=["Classification", "Predicted Classification"])

            st.write(data_display)

        # Questionnaire UI
        st.sidebar.title("Make Diagnosis")
        age = st.sidebar.slider("Age", 0, 100)
        bmi = st.sidebar.number_input("BMI (kg/m2)", min_value=0.0, max_value=50.0, step=0.01)
        glucose = st.sidebar.slider("Glucose (mg/dL)", 0, 200)
        insulin = st.sidebar.number_input("Insulin (µU/mL)", min_value=0.0, max_value=400.0, step=0.01)
        homa = st.sidebar.number_input("HOMA", min_value=0.0, max_value=10.0, step=0.01)
        leptin = st.sidebar.number_input("Leptin (ng/mL)", min_value=0.0, max_value=100.0, step=0.01)
        adiponectin = st.sidebar.number_input("Adiponectin (µg/mL)", min_value=0.0, max_value=50.0, step=0.01)
        resistin = st.sidebar.number_input("Resistin (ng/mL)", min_value=0.0, max_value=50.0, step=0.01)
        mcp1 = st.sidebar.number_input("MCP-1 (pg/dL)", min_value=0.0, max_value=2000.0, step=0.01)

        if st.sidebar.button("Predict"):
            if clf is None:
                # Trained Model Loading (Helps with performance)
                clf = joblib.load('trained_model.joblib')

            if clf is not None:
                # Questionnaire Data Input Preparation
                user_input = pd.DataFrame({
                    "Age": [age],
                    "BMI": [bmi],
                    "Glucose": [glucose],
                    "Insulin": [insulin],
                    "HOMA": [homa],
                    "Leptin": [leptin],
                    "Adiponectin": [adiponectin],
                    "Resistin": [resistin],
                    "MCP.1": [mcp1]
                })

                # Prediction
                prediction = clf.predict(user_input)

                st.write("## Prediction Results")
                st.write(f"Algorithm predicts Classification: {prediction[0]}")
            else:
                st.write("Please train a model before making predictions.")

if __name__ == "__main__":
    main()
