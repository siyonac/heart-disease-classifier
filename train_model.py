import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Column names from the UCI Heart Disease dataset
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]


# Load the dataset
df = pd.read_csv(
    "data/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)


# Separate features and target
X = df.drop("target", axis=1)

# Convert target into binary classification:
# 0 = no significant heart disease
# 1 = heart disease present
y = (df["target"] > 0).astype(int)


print("Dataset shape:", df.shape)
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# Decision Tree pipeline
dt_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ))
])


# Random Forest pipeline
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=50,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42
    ))
])


# Train both models
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

print("\nModels trained successfully!")


# Test predictions
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)


# Test accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nDecision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)


# Random Forest confusion matrix
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# Random Forest classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))


# 5-fold cross-validation
dt_scores = cross_val_score(
    dt_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

rf_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)


print("\n5-Fold Cross-Validation Results")
print("Decision Tree Average:", dt_scores.mean())
print("Random Forest Average:", rf_scores.mean())


import joblib

# Save the trained Random Forest pipeline
joblib.dump(rf_model, "model.pkl")

print("\nRandom Forest model saved as model.pkl")