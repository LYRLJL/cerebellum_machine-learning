from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Data loading
data = pd.read_excel(r'D:')

# Data cleaning and preparation
data_cleaned = data.drop(['File Name'], axis=1).dropna()
X = data_cleaned.drop(['Target'], axis=1)
label_mapping = {'EMCI': 0, 'LMCI': 1}  # Mapping target labels
y = data_cleaned['Target'].map(label_mapping)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_score):
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # Recall (Sensitivity)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # AUC
    auc = roc_auc_score(y_true, y_score)

    return accuracy, sensitivity, specificity, auc

# Create KNN classifier object
knn = KNeighborsClassifier()

# Define the parameter grid for grid search
param_grid = {
    "n_neighbors": np.arange(1,11),
}

# Create GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Perform grid search on the training set
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters from grid search:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the training and test sets
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Get predicted probabilities
y_proba_train = best_model.predict_proba(X_train)[:, 1]  # Probability of the positive class
y_proba_test = best_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Calculate evaluation metrics
train_metrics = calculate_metrics(y_train, y_pred_train, y_proba_train)
test_metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)

# Output training and test metrics
print("\nTraining set metrics:")
print(
    f"Accuracy: {train_metrics[0]:.3f}, Sensitivity (Recall): {train_metrics[1]:.3f}, Specificity: {train_metrics[2]:.3f}, AUC: {train_metrics[3]:.3f}")

print("\nTest set metrics:")
print(
    f"Accuracy: {test_metrics[0]:.3f}, Sensitivity (Recall): {test_metrics[1]:.3f}, Specificity: {test_metrics[2]:.3f}, AUC: {test_metrics[3]:.3f}")

