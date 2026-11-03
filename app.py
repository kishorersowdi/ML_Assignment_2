
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("Online Shoppers Purchase Intention")
st.write("Evaluate five machine learning models using uploaded test data.")

model_options = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

selected_model = st.selectbox(
    "Select a Classification Model",
    list(model_options.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write("Dataset shape:", test_data.shape)
    st.dataframe(test_data.head())

    target_column = "Revenue"

    if target_column not in test_data.columns:
        st.error("The uploaded CSV must contain the Revenue target column.")

    else:
        X_test_app = test_data.drop(columns=[target_column])
        y_test_app = test_data[target_column]

        expected_features = [
            'Administrative', 'Administrative_Duration',
            'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues',
            'SpecialDay', 'Month', 'OperatingSystems',
            'Browser', 'Region', 'TrafficType',
            'VisitorType', 'Weekend'
        ]

        if set(X_test_app.columns) != set(expected_features):
            st.error("Uploaded CSV does not contain the expected 17 features.")

        elif X_test_app.isnull().any().any() or y_test_app.isnull().any():
            st.error("Uploaded CSV contains missing values.")

        elif y_test_app.nunique() != 2:
            st.error("Revenue must contain both target classes.")

        else:
            # Arrange features in the expected order
            X_test_app = X_test_app[expected_features]

            model_path = os.path.join(
                "model", model_options[selected_model]
            )

            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")

            else:
                model = joblib.load(model_path)

                y_pred = model.predict(X_test_app)
                y_prob = model.predict_proba(X_test_app)[:, 1]

                metrics = {
                    "Accuracy": accuracy_score(y_test_app, y_pred),
                    "AUC": roc_auc_score(y_test_app, y_prob),
                    "Precision": precision_score(
                        y_test_app, y_pred, pos_label=True,
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test_app, y_pred, pos_label=True,
                        zero_division=0
                    ),
                    "F1 Score": f1_score(
                        y_test_app, y_pred, pos_label=True,
                        zero_division=0
                    ),
                    "MCC": matthews_corrcoef(y_test_app, y_pred)
                }

                st.subheader("Model Evaluation Results")
                st.write("Selected Model:", selected_model)

                cols = st.columns(3)
                for col, (name, value) in zip(cols, list(metrics.items())[:3]):
                    col.metric(name, f"{value:.4f}")

                cols = st.columns(3)
                for col, (name, value) in zip(cols, list(metrics.items())[3:]):
                    col.metric(name, f"{value:.4f}")

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test_app, y_pred, labels=[False, True]
                )

                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Purchase", "Purchase"],
                    yticklabels=["No Purchase", "Purchase"],
                    ax=ax
                )

                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Classification Report")

                report = classification_report(
                    y_test_app, y_pred,
                    labels=[False, True],
                    target_names=["No Purchase", "Purchase"],
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(
                    pd.DataFrame(report).transpose().round(4)
                )

else:
    st.info("Please upload a test CSV file to evaluate a model.")
