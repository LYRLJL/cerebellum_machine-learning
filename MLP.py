import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Read data
data_path = r"D:\"
data = pd.read_excel(data_path)

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


# Define the parameter grid for MLPClassifier
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  # Different architectures for hidden layers
    'activation': ['tanh', 'relu'],  # Activation functions
    'solver': ['lbfgs', 'sgd', 'adam'],  # Solvers
    'alpha': [0.0001, 0.001, 0.01],  # Regularization term
    'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate schedules
    'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
    'max_iter': [200, 500, 1000]  # Maximum number of iterations
}

# Initialize the MLPClassifier
mlp = MLPClassifier(random_state=10)

# Create GridSearchCV object for parameter search
grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=5, verbose=3, n_jobs=-1, scoring='accuracy')

# Perform GridSearchCV
grid_search_mlp.fit(X_train, y_train)

# Get the best parameters
best_mlp_params = grid_search_mlp.best_params_
print("Best parameters for MLP model:", best_mlp_params)

# Get the best model
best_mlp_model = grid_search_mlp.best_estimator_

# Predict on the training and test sets
y_pred_train_mlp = best_mlp_model.predict(X_train)
y_pred_test_mlp = best_mlp_model.predict(X_test)

# Get predicted probabilities for AUC calculation
y_proba_train_mlp = best_mlp_model.predict_proba(X_train)[:, 1]
y_proba_test_mlp = best_mlp_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics (Accuracy, Sensitivity (Recall), Specificity, AUC)
train_metrics_mlp = calculate_metrics(y_train, y_pred_train_mlp, y_proba_train_mlp)
test_metrics_mlp = calculate_metrics(y_test, y_pred_test_mlp, y_proba_test_mlp)

# Print results
print("\nTraining set metrics for MLP:")
print(
    f"Accuracy: {train_metrics_mlp[0]:.3f}, Sensitivity (Recall): {train_metrics_mlp[1]:.3f}, Specificity: {train_metrics_mlp[2]:.3f}, AUC: {train_metrics_mlp[3]:.3f}")

print("\nTest set metrics for MLP:")
print(
    f"Accuracy: {test_metrics_mlp[0]:.3f}, Sensitivity (Recall): {test_metrics_mlp[1]:.3f}, Specificity: {test_metrics_mlp[2]:.3f}, AUC: {test_metrics_mlp[3]:.3f}")

# Calculate 95% Confidence Intervals for the metrics
ci_train_mlp = bootstrap_ci_from_preds(y_train, y_pred_train_mlp, y_proba_train_mlp)
ci_test_mlp = bootstrap_ci_from_preds(y_test, y_pred_test_mlp, y_proba_test_mlp)

# Output 95% CI results for MLP
print("\nTraining set 95% confidence intervals for MLP:")
print(
    f"Accuracy: {ci_train_mlp['acc']}, AUC: {ci_train_mlp['auc']}, Sensitivity (Recall): {ci_train_mlp['sens']}, Specificity: {ci_train_mlp['spec']}")

print("\nTest set 95% confidence intervals for MLP:")
print(
    f"Accuracy: {ci_test_mlp['acc']}, AUC: {ci_test_mlp['auc']}, Sensitivity (Recall): {ci_test_mlp['sens']}, Specificity: {ci_test_mlp['spec']}")

