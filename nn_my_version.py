import numpy as np
import pandas as pd
import uproot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import joblib
import optuna
from scipy.optimize import minimize_scalar

# 1. Load Data from ROOT Files with All Branches
data_file = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_data_events.root'
mc_file = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_mc_events.root'
tree_name = 'tree'  # Change to the name of your TTree

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
# X.loc[:, 'eta_phi_diff'] = X['eta_1'] - X['phi_1']
# X.loc[:, 'pt_eta_product'] = X['pt_1'] * X['eta_1']
# X.loc[:, 'pt_squared'] = X['pt_1'] ** 2

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the test data to a ROOT file without the label
test_data = X_all.loc[X_test.index].copy()
test_data['label'] = y_test

# Function to apply temperature scaling for binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apply_temperature_scaling_binary(logits, temperature):
    scaled_logits = logits / temperature
    return sigmoid(scaled_logits)

# 4. Hyperparameter Optimization with Optuna
def create_model(trial):
    model = Sequential()
    model.add(Dense(trial.suggest_int('units_layer_1', 32, 128), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(trial.suggest_float('dropout_layer_1', 0.2, 0.5)))
    model.add(Dense(trial.suggest_int('units_layer_2', 32, 128), activation='relu'))
    model.add(Dropout(trial.suggest_float('dropout_layer_2', 0.2, 0.5)))
    model.add(Dense(1, activation='sigmoid'))
    
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['AUC'])
    
    return model

def objective(trial):
    model = create_model(trial)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    
    y_pred_proba = model.predict(X_test).ravel()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return roc_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)

# 5. K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
roc_aucs = []
train_losses = []
val_losses = []

for train_index, test_index in kf.split(X):
    X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
    y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

    # Train the model with the best hyperparameters
    model = Sequential()
    model.add(Dense(best_params['units_layer_1'], activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(best_params['dropout_layer_1']))
    model.add(Dense(best_params['units_layer_2'], activation='relu'))
    model.add(Dropout(best_params['dropout_layer_2']))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='binary_crossentropy', metrics=['AUC'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train_kf, y_train_kf, validation_data=(X_test_kf, y_test_kf), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])
    
    y_pred_proba_kf = model.predict(X_test_kf).ravel()
    roc_auc = roc_auc_score(y_test_kf, y_pred_proba_kf)
    roc_aucs.append(roc_auc)

print("Average ROC AUC Score from K-Folds: ", np.mean(roc_aucs))

# Plotting K-Fold Loss Curves
plt.figure(figsize=(10, 6))
for i in range(len(train_losses)):
    plt.plot(train_losses[i], label=f'Train Loss Fold {i+1}', linestyle='--')
    plt.plot(val_losses[i], label=f'Val Loss Fold {i+1}', linestyle='-')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("K-Fold Training and Validation Loss Curves")
plt.legend()
plt.savefig("best_models_nn/best_model_tau1_var_kfold_loss_curve.pdf")

# Train the final model with the best hyperparameters
model = Sequential()
model.add(Dense(best_params['units_layer_1'], activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(best_params['dropout_layer_1']))
model.add(Dense(best_params['units_layer_2'], activation='relu'))
model.add(Dropout(best_params['dropout_layer_2']))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='binary_crossentropy', metrics=['AUC'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

# Save the best model to a .keras file
model.save('best_models_nn/best_model_tau1_var.keras')

# 6. Plot the Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label="Training Loss", color='blue')
plt.plot(history.history['val_loss'], label="Validation Loss", color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.savefig("best_models_nn/best_model_tau1_var_loss.pdf")

# 7. Evaluate the Model
y_pred_proba = model.predict(X_test).ravel()
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
logits = model.predict(X_test).ravel()

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
plt.savefig("best_models_nn/best_model_tau1_var_roc_curve.pdf")  # Save the ROC curve plot

# Plot and save the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_temp, tpr_temp, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc_temp_scaled:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("best_models_nn/best_model_tau1_var_roc_curve_temp_scaled.pdf")  # Save the ROC curve plot