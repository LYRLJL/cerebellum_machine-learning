import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore', category=UserWarning)  # Ignore warnings

# Data loading
data = pd.read_excel(r'D:\Multimodel_EF_AR\小脑数据分类\网络特征分类\第二次\EMCI_LMCI_ML_T检验.xlsx')

# Data cleaning and preparation
data_cleaned = data.drop(['File Name'], axis=1).dropna()
X = data_cleaned.drop(['Target'], axis=1)
label_mapping = {'EMCI': 0, 'LMCI': 1}  # Mapping target labels
y = data_cleaned['Target'].map(label_mapping)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


def calculate_metrics_two(y_true, y_pred, y_score):
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # Recall (Sensitivity)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # AUC
    auc = roc_auc_score(y_true, y_score)

    return accuracy, sensitivity, specificity, auc


# Create logistic regression model object
logistic_regression = LogisticRegression()

# Define parameter grid to search
param_grid = {
    'max_iter': [500, 1000, 2000],  # Grid for max_iter
    'random_state': [10, 42, 123],  # Grid for random_stat
    'penalty': ['l2',None]  # Regularization types: L1 or L2
}

# Create grid search object
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, verbose=3)

# Perform grid search
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print("Best parameters from grid search:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Train and predict
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Get predicted probabilities
y_proba_train = best_model.predict_proba(X_train)[:, 1]  # Probability of the positive class
y_proba_test = best_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Calculate evaluation metrics
train_metrics = calculate_metrics_two(y_train, y_pred_train, y_proba_train)
test_metrics = calculate_metrics_two(y_test, y_pred_test, y_proba_test)

# Output training and test metrics
print("\nTraining set metrics:")
print(
    f"Accuracy: {train_metrics[0]:.3f}, Sensitivity (Recall): {train_metrics[1]:.3f}, Specificity: {train_metrics[2]:.3f}, AUC: {train_metrics[3]:.3f}")

print("\nTest set metrics:")
print(
    f"Accuracy: {test_metrics[0]:.3f}, Sensitivity (Recall): {test_metrics[1]:.3f}, Specificity: {test_metrics[2]:.3f}, AUC: {test_metrics[3]:.3f}")


