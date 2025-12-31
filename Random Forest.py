# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import warnings
from scipy.stats import randint
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore")

# Create folder to save charts
output_dir = "plots_and_tables"
os.makedirs(output_dir, exist_ok=True)

# Read data
data_path = r"D:"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

data = pd.read_excel(data_path)

# Data cleaning and preparation
data_cleaned = data.drop(['File Name'], axis=1).dropna()
X = data_cleaned.drop(['Target'], axis=1)
y = data_cleaned['Target']

# Encode the target variable as numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Print feature count for debugging
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")
print(f"Target variable shape: {y.shape}")
print(f"Unique values in target variable: {y.unique()}")

# Ensure it's a binary classification problem
if len(np.unique(y)) != 2:
    raise ValueError("Target variable is not binary, please check the data!")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# Get label names
labels = label_encoder.classes_

# Define hyperparameter search space
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Hyperparameter search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring="accuracy"
)

# Train the model
print("Starting Randomized Search...")
random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(random_search.best_score_))

# Best model
best_rf = random_search.best_estimator_

# Evaluate the model
test_accuracy = best_rf.score(X_test, y_test)
print(f"Test set accuracy: {test_accuracy:.2f}")

# Confusion matrix and classification report
y_pred = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)


