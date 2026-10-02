# Heart Disease Classifier

A machine learning project that uses the UCI Heart Disease dataset to classify whether a patient record indicates the presence of significant heart disease.

## Features

- Data preprocessing and missing-value handling
- Decision Tree classification
- Random Forest classification
- 5-fold cross-validation
- Confusion matrix and classification report
- Interactive Streamlit web application
- Saved trained Random Forest model

## Machine Learning Results

Using an 80/20 train-test split:

| Model         | Test Accuracy | 5-Fold CV Average |
-----------------------------------------------------
| Decision Tree | 78.69%        | 73.52% |
| Random Forest | 90.16%        | 81.38% |

The Random Forest was used for the interactive application.

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- joblib
- Git/GitHub

## Running the Project

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt