import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load data
file_path = r"D:\Multimodel_EF_AR\小脑数据分类\网络特征分类\第二次\EMCI_LMCI_ML_T检验.xlsx"
data = pd.read_excel(file_path)

# Data cleaning and preparation
data_cleaned = data.drop(['File Name'], axis=1).dropna()
X = data_cleaned.drop(['Target'], axis=1)
y = data_cleaned['Target']

# Encode the target variable as numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=10)


# Function to calculate metrics (Accuracy, Sensitivity, Specificity, AUC)
def calculate_metrics(y_true, y_pred, y_score):
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # Sensitivity (Recall)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # AUC
    auc = roc_auc_score(y_true, y_score)

    return accuracy, sensitivity, specificity, auc


# Function to calculate 95% Confidence Interval using Bootstrap
def bootstrap_ci_from_preds(y_true, y_pred, y_score, n_iter=2000, seed=10):
    """Bootstrap to calculate the 95% confidence intervals for the metrics"""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    accs, aucs, senss, specs = [], [], [], []

    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)  # Sample indices with replacement
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        accuracy, sensitivity, specificity, auc_val = calculate_metrics(yt, yp, ys)
        accs.append(accuracy);
        aucs.append(auc_val);
        senss.append(sensitivity);
        specs.append(specificity)

    def ci(arr):
        return (np.round(np.percentile(arr, 2.5), 3), np.round(np.percentile(arr, 97.5), 3)) if len(arr) else (
        np.nan, np.nan)

    return {"acc": ci(accs), "auc": ci(aucs), "sens": ci(senss), "spec": ci(specs)}


# Define the parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],  # Number of boosting rounds
    'max_depth': [3, 6, 10],  # Maximum depth of the trees
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples to use for each boosting round
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features to use for each tree
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction to make a further partition
    'reg_alpha': [0, 0.1, 0.2],  # L1 regularization term
    'reg_lambda': [0.1, 1, 10]  # L2 regularization term
}

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=10)

# Create GridSearchCV object for parameter search
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, verbose=3, n_jobs=-1,
                               scoring='accuracy')

# Perform GridSearchCV
grid_search_xgb.fit(X_train, y_train)

# Get the best parameters
best_xgb_params = grid_search_xgb.best_params_
print("Best parameters for XGBoost model:", best_xgb_params)

# Get the best model
best_xgb_model = grid_search_xgb.best_estimator_

# Predict on the training and test sets
y_pred_train_xgb = best_xgb_model.predict(X_train)
y_pred_test_xgb = best_xgb_model.predict(X_test)

# Get predicted probabilities for AUC calculation
y_proba_train_xgb = best_xgb_model.predict_proba(X_train)[:, 1]
y_proba_test_xgb = best_xgb_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics (Accuracy, Sensitivity (Recall), Specificity, AUC)
train_metrics_xgb = calculate_metrics(y_train, y_pred_train_xgb, y_proba_train_xgb)
test_metrics_xgb = calculate_metrics(y_test, y_pred_test_xgb, y_proba_test_xgb)

# Print results
print("\nTraining set metrics for XGBoost:")
print(
    f"Accuracy: {train_metrics_xgb[0]:.3f}, Sensitivity (Recall): {train_metrics_xgb[1]:.3f}, Specificity: {train_metrics_xgb[2]:.3f}, AUC: {train_metrics_xgb[3]:.3f}")

print("\nTest set metrics for XGBoost:")
print(
    f"Accuracy: {test_metrics_xgb[0]:.3f}, Sensitivity (Recall): {test_metrics_xgb[1]:.3f}, Specificity: {test_metrics_xgb[2]:.3f}, AUC: {test_metrics_xgb[3]:.3f}")

# Calculate 95% Confidence Intervals for the metrics
ci_train_xgb = bootstrap_ci_from_preds(y_train, y_pred_train_xgb, y_proba_train_xgb)
ci_test_xgb = bootstrap_ci_from_preds(y_test, y_pred_test_xgb, y_proba_test_xgb)

# Output 95% CI results for XGBoost
print("\nTraining set 95% confidence intervals for XGBoost:")
print(
    f"Accuracy: {ci_train_xgb['acc']}, AUC: {ci_train_xgb['auc']}, Sensitivity (Recall): {ci_train_xgb['sens']}, Specificity: {ci_train_xgb['spec']}")

print("\nTest set 95% confidence intervals for XGBoost:")
print(
    f"Accuracy: {ci_test_xgb['acc']}, AUC: {ci_test_xgb['auc']}, Sensitivity (Recall): {ci_test_xgb['sens']}, Specificity: {ci_test_xgb['spec']}")
