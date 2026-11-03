# Online Shoppers Purchasing Intention – Machine Learning Assignment 2

**BITS Pilani – M.Tech Artificial Intelligence and Machine Learning**  
**Course:** Machine Learning  
**Assignment:** Assignment 2  
**Dataset:** Online Shoppers Purchasing Intention Dataset

---

## 1. Problem Statement

The objective of this assignment is to develop and evaluate multiple machine learning classification models to predict whether an online shopping session will result in a purchase.

Understanding purchase intention can help e-commerce businesses identify potential customers, improve conversion rates, and optimize their marketing strategies.

This project implements five supervised machine learning classification algorithms and compares their performance using multiple evaluation metrics.

The five models implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset size:** 12,330 instances  
**Total columns:** 18 (17 input features and 1 target variable)  
**Problem type:** Binary Classification  
**Target variable:** `Revenue`

The dataset contains information about online shopping sessions, including visitor activity, browsing behavior, page duration, traffic source, operating system, browser, and visitor type.

### Target Variable

The target variable `Revenue` indicates whether a purchase was made during the session:

- `True`: Purchase completed.
- `False`: No purchase completed.

### Feature Categories

**Numerical features (10):**
- Administrative
- Administrative_Duration
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- SpecialDay

**Categorical features (7):**
- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType
- Weekend

### Class Distribution

The dataset exhibits class imbalance:

- No Purchase (`False`): 10,422 instances (approximately 84.5%).
- Purchase (`True`): 1,908 instances (approximately 15.5%).

This imbalance makes it important to evaluate models using metrics beyond accuracy.

## 3. Data Preprocessing and Model Implementation

The following preprocessing and modelling steps were performed:

1. Loaded and explored the dataset.
2. Examined the target distribution and feature characteristics.
3. Separated the independent features (`X`) and target variable (`y`).
4. Split the dataset into training and testing sets using an 80:20 ratio.
5. Used stratified sampling to preserve the target class distribution across both sets.
6. Applied StandardScaler to numerical features.
7. Applied OneHotEncoder to categorical features.
8. Used Scikit-learn Pipeline and ColumnTransformer to combine preprocessing and classification.

The preprocessing transformations were fitted on the training data and applied to the test data through the model pipelines to avoid data leakage.

**Train-test split:**
- Training instances: 9,864
- Testing instances: 2,466

### Models Implemented

| Model | Implementation |
|---|---|
| Logistic Regression | LogisticRegression |
| Decision Tree | DecisionTreeClassifier |
| KNN | KNeighborsClassifier (k=5) |
| Naive Bayes | GaussianNB |
| SVM | SVC (RBF kernel, probability enabled) |

All five trained pipelines, including their preprocessing transformations, are saved as Joblib artifacts in the `model/` directory.

## 4. Model Performance Comparison

All five models were evaluated on the same held-out test dataset containing 2,466 instances.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8585 | 0.7356 | 0.5420 | 0.5576 | 0.5497 | 0.4658 |
| KNN | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| SVM | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |

## 5. Model-wise Observations

### 5.1 Logistic Regression

- Achieved an accuracy of 88.08%.
- Recorded the highest AUC (0.8828) and precision (0.7366).
- Demonstrated good ability to distinguish between purchasing and non-purchasing sessions.
- However, recall was relatively low at 0.3586, indicating that many actual purchase sessions were missed.

**Observation:** Logistic Regression provides strong ranking and precision performance but is relatively conservative in identifying purchase sessions.

### 5.2 Decision Tree Classifier

- Achieved an accuracy of 85.85%.
- Recorded a recall of 0.5576, the second-highest among the five models.
- Achieved an F1 Score of 0.5497.
- Its AUC of 0.7356 was lower than those of Logistic Regression and SVM.

**Observation:** Decision Tree identifies more actual purchasing sessions than Logistic Regression and KNN, but produces more false positives and has weaker overall discrimination.

### 5.3 K-Nearest Neighbors (KNN)

- Achieved an accuracy of 87.67%.
- Recorded precision of 0.6822 and recall of 0.3822.
- Achieved an F1 Score of 0.4899.
- Its AUC was 0.7633.

**Observation:** KNN demonstrates reasonable overall accuracy but has limited ability to identify actual purchasing sessions.

### 5.4 Gaussian Naive Bayes

- Achieved an accuracy of 27.29%.
- Recorded the highest recall of 0.9738.
- However, precision was only 0.1726.
- Produced a large number of false positives, resulting in a low MCC of 0.1289.

**Observation:** Naive Bayes identifies almost all actual purchase sessions but incorrectly classifies a large number of non-purchasing sessions as purchases. Its overall performance is poor for this dataset.

### 5.5 Support Vector Machine (SVM)

- Achieved the highest accuracy of 88.73%.
- Recorded the highest F1 Score of 0.5629.
- Achieved the highest MCC of 0.5149.
- Its AUC of 0.8824 was very close to Logistic Regression.
- Recorded a recall of 0.4686 and precision of 0.7047.

**Observation:** SVM provides the best overall balance among the evaluated models, with the strongest accuracy, F1 Score, and MCC.

## 6. Overall Model Comparison and Conclusion

Based on the experimental results, **Support Vector Machine (SVM) is the overall best-performing model** among the five implemented classifiers.

It achieved:

- Accuracy: 88.73%
- AUC: 0.8824
- Precision: 0.7047
- Recall: 0.4686
- F1 Score: 0.5629
- MCC: 0.5149

Although Logistic Regression achieved the highest AUC and precision, SVM demonstrated a better balance between precision and recall, resulting in the highest F1 Score and MCC.

The results also highlight the impact of class imbalance. Accuracy alone is not sufficient to assess classification performance, particularly when the objective is to identify the minority purchase class.

Therefore, considering multiple evaluation metrics, SVM is selected as the overall winner for this baseline model comparison.

## 7. Streamlit Application

A Streamlit application has been developed to evaluate the trained classification models using uploaded test data.

### Application Features

- Upload test data in CSV format.
- Select one of the five trained classification models.
- Load the corresponding saved model without retraining.
- Generate predictions on the uploaded dataset.
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Display the confusion matrix.
- Display the classification report.

### Application Link

**Live Streamlit Application:** To be updated after Streamlit Community Cloud deployment.

## 8. GitHub Repository

**Repository:** https://github.com/kishorersowdi/ML_Assignment_2

### Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── 2026AJ05785_KishoreSowdi_Assignment_2.ipynb
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 9. How to Run the Application Locally

### Prerequisites

Python 3.10 or a compatible Python environment.

### Installation

Clone the repository:

```bash
git clone https://github.com/kishorersowdi/ML_Assignment_2.git
cd ML_Assignment_2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in a browser.

Upload a test CSV containing the 17 input features and the `Revenue` target column to evaluate the selected model.

---

**Note:** The reported results represent baseline model performance using the specified train-test split and model configurations. No hyperparameter tuning was performed for this comparison.
