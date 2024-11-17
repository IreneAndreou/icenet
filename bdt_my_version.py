import os
import argparse
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import joblib
import optuna
from scipy.optimize import minimize_scalar

# 0. Parse Arguments
parser = argparse.ArgumentParser(description='Train a BDT model for jet->tau FFs')
parser.add_argument('--era', type=str, choices=['Run3_2022', 'Run3_2022EE'], required=True, help='Era to process: Run3_2022 or Run3_2022EE')
parser.add_argument('--tau', type=str, choices=['leading', 'subleading'], required=True, help='Tau to process: leading or subleading')
args = parser.parse_args()

# Set paths based on the era
# 1. Load Data from ROOT Files with All Branches
tau_suffix = "lead" if args.tau == "leading" else "sublead"
if args.era == 'Run3_2022':
    data_file = f'/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_data_events_{tau_suffix}_Run3_2022.root'
    mc_file = f'/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_mc_events_{tau_suffix}_Run3_2022.root'
    output_dir = f"best_models/Run3_2022/{tau_suffix}/no_global"
elif args.era == 'Run3_2022EE':
    data_file = f'/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_data_events_{tau_suffix}_Run3_2022EE.root'
    mc_file = f'/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_mc_events_{tau_suffix}_Run3_2022EE.root'
    output_dir = f"best_models/Run3_2022EE/{tau_suffix}"

tree_name = 'tree'  # Change to the name of your TTree

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load all branches into DataFrames
data_df = uproot.open(data_file)[tree_name].arrays(library="pd")
mc_df = uproot.open(mc_file)[tree_name].arrays(library="pd")

# 2. Label Data and Combine
data_df['label'] = 1
mc_df['label'] = 0

# Concatenate and shuffle data
combined_df = pd.concat([data_df, mc_df]).sample(frac=1).reset_index(drop=True)

# Separate features and labels
X_all = combined_df.drop(columns=['label'])
y = combined_df['label']

# Select specific branches for training
training_branches = ['decayMode_1', 'jpt_pt_1', 'pt_1', 'eta_1', 'charge_1', 'phi_1', 'decayModePNet_1']
X = X_all[training_branches]

# Feature Engineering: Add new features or transform existing ones
# X = X.copy()
# X['eta_phi_diff'] = X['eta_1'] - X['phi_1']
# X['pt_eta_product'] = X['pt_1'] * X['eta_1']
# X['pt_squared'] = X['pt_1'] ** 2

# Normalize the data -- TODO: Apply to predictions before making predictions
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Convert X back to DataFrame
# new_features = ['eta_phi_diff', 'pt_eta_product', 'pt_squared']
#all_features = training_branches + new_features
all_features = training_branches
X = pd.DataFrame(X, columns=all_features)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Assuming X_test and y_test are your test data and labels
# test_data = X_all.loc[X_test.index].copy()
# test_data['label'] = y_test

# # Save each test sample to a separate ROOT file
# for idx, row in test_data.iterrows():
#     file_name = f"test_sample_{idx}.root"
#     with uproot.recreate(file_name) as f:
#         f["tree"] = uproot.newtree({col: "float64" for col in test_data.columns})
#         f["tree"].extend({col: [row[col]] for col in test_data.columns})

# Function to apply temperature scaling for binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apply_temperature_scaling_binary(logits, temperature):
    scaled_logits = logits / temperature
    return sigmoid(scaled_logits)

# 4. Hyperparameter Optimization with Optuna
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),  # L2 regularization
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'nthread': 8,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    evals_result = {}
    bst = xgb.train(param, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=10, evals_result=evals_result, verbose_eval=False)
    y_pred_proba = bst.predict(dtest)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return roc_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=8)

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)
best_params['eval_metric'] = 'logloss'
best_params['objective'] = 'binary:logistic'
best_params['nthread'] = 8

# 5. K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
roc_aucs = []
train_losses = []
val_losses = []

for train_index, test_index in kf.split(X):
    X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
    y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

    # Train the model with the best hyperparameters
    dtrain = xgb.DMatrix(X_train_kf, label=y_train_kf)
    dtest = xgb.DMatrix(X_test_kf, label=y_test_kf)
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    evals_result = {}
    bst = xgb.train(best_params, dtrain, num_boost_round=1000, evals=evals,
                    early_stopping_rounds=10, evals_result=evals_result, verbose_eval=False)

    # Store training and validation losses
    train_losses.append(evals_result['train']['logloss'])
    val_losses.append(evals_result['eval']['logloss'])

    y_pred_proba_kf = bst.predict(dtest)
    roc_auc = roc_auc_score(y_test_kf, y_pred_proba_kf)
    roc_aucs.append(roc_auc)

print("Average ROC AUC Score from K-Folds: ", np.mean(roc_aucs))

# Plotting K-Fold Loss Curves
plt.figure(figsize=(10, 6))
for i in range(len(train_losses)):
    plt.plot(train_losses[i], label=f'Train Loss Fold {i+1}', linestyle='--')
    plt.plot(val_losses[i], label=f'Val Loss Fold {i+1}', linestyle='-')

plt.xlabel("Boosting Rounds")
plt.ylabel("Log-Loss")
plt.title("K-Fold Training and Validation Loss Curves")
plt.legend()
plt.savefig(f"{output_dir}/best_model_tau1_all_var_kfold_loss_curve.pdf")

# Train the final model with the best hyperparameters
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
evals = [(dtrain, 'train'), (dtest, 'eval')]

evals_result = {}
bst = xgb.train(best_params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=10, evals_result=evals_result, verbose_eval=True)

# Save the best model to a .pkl file
bst.save_model(f"{output_dir}/best_model_tau1_all_var.json")

# Save the best model to a .pkl file
joblib.dump(bst, f"{output_dir}/best_model_tau1_all_var.pkl")

# 6. Plot the Loss Curve
# Print the evals_result dictionary to debug
print("evals_result:", evals_result)

# Retrieve the training and validation loss from evals_result
train_log_loss = evals_result['train']['logloss']
val_log_loss = evals_result['eval']['logloss']

plt.figure(figsize=(10, 6))
plt.plot(train_log_loss, label="Training Log-Loss", color='blue')
plt.plot(val_log_loss, label="Validation Log-Loss", color='orange')
plt.xlabel("Boosting Rounds")
plt.ylabel("Log-Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.savefig(f"{output_dir}/best_model_tau1_all_var_loss.pdf")

# 7. Evaluate the Model
y_pred_proba = bst.predict(dtest)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Original ROC AUC Score: {roc_auc:.4f}")

# Function to find the optimal temperature
def find_optimal_temperature(logits, y):
    def temperature_obj(t):
        temp_logits = logits / t
        temp_probs = sigmoid(temp_logits)
        return log_loss(y, temp_probs)
    
    res = minimize_scalar(temperature_obj, bounds=(1e-2, 100), method='bounded')
    return res.x

# Get logits for the test set
logits = bst.predict(dtest, output_margin=True)

# Find the optimal temperature
optimal_temperature = find_optimal_temperature(logits, y_test)
print(f"Optimal Temperature: {optimal_temperature}")

# Calibrate using Temperature Scaling
y_pred_probs_temp_scaled = apply_temperature_scaling_binary(logits, optimal_temperature)

# Calculate ROC AUC score after temperature scaling
roc_auc_temp_scaled = roc_auc_score(y_test, y_pred_probs_temp_scaled)
print(f"Temperature Scaled ROC AUC Score: {roc_auc_temp_scaled:.4f}")

# Calculate ROC curve points
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Calculate ROC curve points
fpr_temp, tpr_temp, _ = roc_curve(y_test, y_pred_probs_temp_scaled)

# Plot and save the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig(f"{output_dir}/best_model_tau1_all_var_roc_curve.pdf")  # Save the ROC curve plot

# Plot and save the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_temp, tpr_temp, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc_temp_scaled:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig(f"{output_dir}/best_model_tau1_all_var_roc_curve_temp_scaled.pdf")  # Save the ROC curve plot

# 8. Plot Feature Importance
plt.figure(figsize=(10, 6))
plot_importance(bst, importance_type='weight')
plt.title("Feature Importance (Weight)")
plt.savefig(f"{output_dir}/best_model_tau1_all_var_feature_importance_weight.pdf")

plt.figure(figsize=(10, 6))
plot_importance(bst, importance_type='gain')
plt.title("Feature Importance (Gain)")
plt.savefig(f"{output_dir}/best_model_tau1_all_var_feature_importance_gain.pdf")

plt.figure(figsize=(10, 6))
plot_importance(bst, importance_type='cover')
plt.title("Feature Importance (Cover)")
plt.savefig(f"{output_dir}/best_model_tau1_all_var_feature_importance_cover.pdf")