import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# 1. Generate synthetic data
np.random.seed(42)
n_samples = int(1e5)
n_features = 1

# Function to generate weights
def generate_weights(n_samples, random_for_all=False):
    if random_for_all:
        return np.random.rand(n_samples)
    else:
        weights = np.ones(n_samples)
        half_samples = n_samples // 20
        weights[:half_samples] = np.random.rand(half_samples)
        np.random.shuffle(weights)
        return np.abs(weights)

# Class 0: iso data with MC with -ve weights flipped to positive
X1 = np.random.normal(0, 1, (n_samples, n_features))
y1 = np.zeros(n_samples)
weights1 = np.ones(n_samples)
#weights1 = generate_weights(n_samples)

# Class 1: anti-iso data with MC with -ve weights flipped to positive
X2 = np.random.normal(2, 1, (int(n_samples*0.7), n_features))
y2 = np.ones(int(n_samples*0.7))
weights2 = np.ones(int(n_samples*0.7))
#weights2 = generate_weights(n_samples)

# Class 2: +ve MC weights with iso selections
X3 = np.random.normal(0.5, 3, (int(n_samples/2), n_features))
y3 = np.ones(int(n_samples/2)) * 2
weights3 = generate_weights(int(n_samples/2), random_for_all=True)

# Class 3: +ve MC weights with anti-iso selections
X4 = np.random.normal(2.5, 3, (int(n_samples/4), n_features))
y4 = np.ones(int(n_samples/4)) * 3
weights4 = generate_weights(int(n_samples/4), random_for_all=True)

# Class 2: +ve MC weights with iso selections
X5 = np.random.normal(0.5, 3, (int(n_samples), n_features))
y5 = np.zeros(int(n_samples))
weights5 = generate_weights(int(n_samples), random_for_all=True)

# Class 3: +ve MC weights with anti-iso selections
X6 = np.random.normal(2.5, 3, (int(n_samples), n_features))
y6 = np.ones(int(n_samples))
weights6 = generate_weights(int(n_samples), random_for_all=True)

# # Normalize weights5 and weights6 to have a mean of 1
# weights5 /= np.mean(weights5)
# weights6 /= np.mean(weights6)


# Update class 0 to include class 2
X1_comb = np.vstack((X1, X5))
y1_comb = np.concatenate((y1, y5))
y1_comb = np.zeros(len(y1_comb))
weights1_comb = np.concatenate((weights1, weights5))

# Update class 1 to include class 3
X2_comb = np.vstack((X2, X6))
y2_comb = np.concatenate((y2, y6))
y2_comb = np.ones(len(y2_comb))
weights2_comb = np.concatenate((weights2, weights6))

# Calculate the effective yields
effective_yield_class_0 = np.sum(weights1_comb)
effective_yield_class_1 = np.sum(weights2_comb)
effective_yield_class_2 = np.sum(weights3)
effective_yield_class_3 = np.sum(weights4)

print("Effective yields before reweighting:")
print(f"Class 0 (Blue): {effective_yield_class_0:.2f}")
print(f"Class 1 (Orange): {effective_yield_class_1:.2f}")
print(f"Class 2 (Green): {effective_yield_class_2:.2f}")
print(f"Class 3 (Red): {effective_yield_class_3:.2f}")


# NOTE: If I mess with the s.d. of the distributions, the weights will be different

# Compute histograms
hist1, bins = np.histogram(X1.flatten(), bins=50, weights=weights1, range=(-10, 10))
hist2, _ = np.histogram(X2.flatten(), bins=bins, weights=weights2)
hist3, _ = np.histogram(X3.flatten(), bins=bins, weights=weights3)
hist4, _ = np.histogram(X4.flatten(), bins=bins, weights=weights4)
hist5, _ = np.histogram(X5.flatten(), bins=bins, weights=weights5)
hist6, _ = np.histogram(X6.flatten(), bins=bins, weights=weights6) 

hist1 = hist1 + hist5
hist2 = hist2 + hist6

# Plotting the distributions
plt.figure(figsize=(12, 8))

plt.hist(bins[:-1], bins=bins, weights=hist1, alpha=0.5, label='Class 0', histtype='stepfilled')
plt.hist(bins[:-1], bins=bins, weights=hist2, alpha=0.5, label='Class 1', histtype='stepfilled')
plt.hist(bins[:-1], bins=bins, weights=hist3, alpha=0.5, label='Class 2', histtype='stepfilled')
plt.hist(bins[:-1], bins=bins, weights=hist4, alpha=0.5, label='Class 3', histtype='stepfilled')
# plt.hist(bins[:-1], bins=bins, weights=hist5, alpha=0.5, label='Class 4', histtype='stepfilled')
# plt.hist(bins[:-1], bins=bins, weights=hist6, alpha=0.5, label='Class 5', histtype='stepfilled')


plt.xlabel('Value')
plt.ylabel('Weighted Count')
plt.title('Distributions with Weights')
plt.legend()
plt.savefig('distributions_with_weights.png')
print('Distributions with weights saved to distributions_with_weights.png')



# Subtract histograms
hist_sub_1_3 = hist1 - (hist3*2)
hist_sub_2_4 = hist2 - (hist4*4)

# Check for negative bins
negative_bins_1_3 = np.any(hist_sub_1_3 < 0)
negative_bins_2_4 = np.any(hist_sub_2_4 < 0)

if negative_bins_1_3:
    print("There are negative bins in hist_sub_1_3")
    print(hist_sub_1_3[np.where(hist_sub_1_3 < 0)])
else:
    print("No negative bins in hist_sub_1_3")

if negative_bins_2_4:
    print("There are negative bins in hist_sub_2_4")
    print(hist2)
    print(hist4)
    print(hist_sub_2_4[np.where(hist_sub_2_4 < 0)])
    print(hist_sub_2_4)
    negative_indices_2_4 = np.where(hist_sub_2_4 < 0)
    print("Indices of negative bins in hist_sub_2_4:", negative_indices_2_4)
else:
    print("No negative bins in hist_sub_2_4")

# Plotting the subtracted histograms
plt.figure(figsize=(12, 8))

plt.hist(bins[:-1], bins=bins, weights=hist_sub_1_3, alpha=0.5, label='Class 0 - Class 1', histtype='stepfilled')
plt.hist(bins[:-1], bins=bins, weights=hist_sub_2_4, alpha=0.5, label='Class 2 - Class 3', histtype='stepfilled')

plt.xlabel('Value')
plt.ylabel('Weighted Count Difference')
plt.title('Subtracted Distributions with Weights')
plt.legend()
plt.savefig('subtracted_distributions_with_weights.png')
print('Subtracted distributions with weights saved to subtracted_distributions_with_weights.png')


# # Find nearest neighbors
# tree_X1 = cKDTree(X1)
# distances, indices = tree_X1.query(X3, k=1)

# # Subtract nearest neighbors
# X1_minus_X3 = np.delete(X1, indices, axis=0)

# weights_X1_minus_X3 = np.delete(weights1, indices, axis=0)

# tree_X2 = cKDTree(X2)
# distances, indices = tree_X2.query(X4, k=1)

# # Subtract nearest neighbors
# X2_minus_X4 = np.delete(X2, indices, axis=0)
# weights_X2_minus_X4 = np.delete(weights2, indices, axis=0)

# # Plotting the distributions
# plt.figure(figsize=(12, 8))

# plt.hist(X1_minus_X3.flatten(), bins=50, weights=weights_X1_minus_X3, alpha=0.5, label='C0 - C2')
# plt.hist(X2_minus_X4.flatten(), bins=50, weights=weights_X2_minus_X4, alpha=0.5, label='C1 - C3')

# plt.xlabel('Value')
# plt.ylabel('Weighted Count')
# plt.title('Distributions of C0 - C2 and C1 - C3 with Weights')
# plt.legend()
# plt.savefig('distributions_C0_minus_C2_C1_minus_C3_with_weights.png')
# plt.show()

# Combine the data
X = np.vstack((X1_comb, X2_comb, X3, X4))
y = np.hstack((y1_comb, y2_comb, y3, y4))
weights = np.hstack((weights1_comb, weights2_comb, weights3, weights4))
print(f"Length of X: {len(X)}")
print(f"Length of y: {len(y)}")
print(f"Length of weights: {len(weights)}")

# 2. Prepare the data for training
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.3, random_state=42)

# 3. Train a multi-class BDT using xgboost
param = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'multi:softprob',
    'num_class': 4,
    'eval_metric': 'mlogloss',
    'nthread': 4,
    'lambda': 1.0  # L2 regularization term

}

# Need to ensure robust weights

dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
dtest = xgb.DMatrix(X_test, label=y_test, weight=weights_test)
evals = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(param, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10, verbose_eval=True)

# 4. Predict the probability scores for each class - reweighting class 1 samples
# y_pred_proba = bst.predict(xgb.DMatrix(X2))
# class_0_proba = y_pred_proba[:, 0]
# class_1_proba = y_pred_proba[:, 1]
# class_2_proba = y_pred_proba[:, 2]
# class_3_proba = y_pred_proba[:, 3]

# #Reweight
# # Derive the new weights for class 1 samples
# reweight = (class_0_proba - class_2_proba) / (class_1_proba - class_3_proba)
# #reweight = class_0_proba / (class_1_proba +class_2_proba + class_3_proba)
# new_weights_class_1 = weights2 * reweight

# # Normalize new weights to sum of old weights
# sum_old_weights = np.sum(weights_X2_minus_X4)
# sum_new_weights = np.sum(new_weights_class_1)
# normalized_new_weights_class_1 = new_weights_class_1 * (sum_old_weights / sum_new_weights)

# Predict the probability scores for each class - reweighting class 1 samples
y_pred_proba_X2 = bst.predict(xgb.DMatrix(X2_comb))
class_0_proba_X2 = y_pred_proba_X2[:, 0]
class_1_proba_X2 = y_pred_proba_X2[:, 1]
class_2_proba_X2 = y_pred_proba_X2[:, 2]
class_3_proba_X2 = y_pred_proba_X2[:, 3]

# Reweight for X2
reweight_X2 = class_0_proba_X2 / class_1_proba_X2
new_weights_class_1_X2 = weights2_comb * reweight_X2

# Normalize new weights to sum of old weights for X2
sum_old_weights_X2 = np.sum(weights1_comb)
sum_new_weights_X2 = np.sum(new_weights_class_1_X2)
normalized_new_weights_class_1_X2 = new_weights_class_1_X2 * (sum_old_weights_X2 / sum_new_weights_X2)

# Predict the probability scores for each class - reweighting class 3 samples
y_pred_proba_X4 = bst.predict(xgb.DMatrix(X4))
class_0_proba_X4 = y_pred_proba_X4[:, 0]
class_1_proba_X4 = y_pred_proba_X4[:, 1]
class_2_proba_X4 = y_pred_proba_X4[:, 2]
class_3_proba_X4 = y_pred_proba_X4[:, 3]

# Reweight for X4
reweight_X4 = class_2_proba_X4 / class_3_proba_X4
new_weights_class_3_X4 = weights4 * reweight_X4

# Normalize new weights to sum of old weights for X4
sum_old_weights_X4 = np.sum(weights3)
sum_new_weights_X4 = np.sum(new_weights_class_3_X4)
normalized_new_weights_class_3_X4 = new_weights_class_3_X4 * (sum_old_weights_X4 / sum_new_weights_X4)


# Compute histograms
hist2_r, _ = np.histogram(X2_comb.flatten(), bins=bins, weights=normalized_new_weights_class_1_X2)
hist4_r, _ = np.histogram(X4.flatten(), bins=bins, weights=normalized_new_weights_class_3_X4)

# Plotting the reweight distribution
plt.figure(figsize=(12, 8))
plt.hist(bins[:-1], bins=bins, weights=hist1, alpha=0.5, label='Class 0', histtype='stepfilled')
plt.hist(bins[:-1], bins=bins, weights=hist2_r, alpha=0.5, label='Reweighted Class 1', histtype='stepfilled')
plt.hist(bins[:-1], bins=bins, weights=hist3, alpha=0.5, label='Class 2', histtype='stepfilled')
plt.hist(bins[:-1], bins=bins, weights=hist4_r, alpha=0.5, label='Reweighted Class 3', histtype='stepfilled')

plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Reweighted Distribution')
plt.legend()
plt.savefig('reweight_distributions.png')
print('Reweighted distributions saved to reweight_distributions.png')

print("\nEffective yields after reweighting:")
print(f"Class 1 (Reweighted Orange): {np.sum(normalized_new_weights_class_1_X2):.2f}")
print(f"Class 3 (Reweighted Red): {np.sum(normalized_new_weights_class_3_X4):.2f}")


# Plotting the subtracted histograms - after reweighting
hist_sub_2_4 = hist2_r - (hist4_r*2)  #needs times two because normalised to class 2
plt.figure(figsize=(12, 8))

plt.hist(bins[:-1], bins=bins, weights=hist_sub_1_3, alpha=0.5, label='Class 0 - Class 1', histtype='stepfilled')
plt.hist(bins[:-1], bins=bins, weights=hist_sub_2_4, alpha=0.5, label='Class 2 - Class 3', histtype='stepfilled')

plt.xlabel('Value')
plt.ylabel('Weighted Count Difference')
plt.title('Subtracted Distributions with Weights - After Reweighting')
plt.legend()
plt.savefig('subtracted_distributions_with_weights_after.png')
print('Subtracted distributions with weights saved to subtracted_distributions_with_weights_after.png')

# # Compute histograms
# hist1, bins = np.histogram(X1.flatten(), bins=50, weights=weights1)
# hist2, _ = np.histogram(X2_comb.flatten(), bins=bins, weights=normalized_new_weights_class_1_X2)
# hist3, _ = np.histogram(X3.flatten(), bins=bins, weights=weights3)
# hist4, _ = np.histogram(X4.flatten(), bins=bins, weights=normalized_new_weights_class_3_X4)

# # Subtract histograms
# hist_sub_1_3 = hist1 - hist3
# hist_sub_2_4 = hist2_r - hist4_r

# # Plotting the subtracted histograms
# plt.figure(figsize=(12, 8))

# plt.bar(bins[:-1], hist_sub_1_3, width=np.diff(bins), alpha=0.5, label='Class 0 - Class 1')
# plt.bar(bins[:-1], hist_sub_2_4, width=np.diff(bins), alpha=0.5, label='Class 2 - Class 3')

# plt.xlabel('Value')
# plt.ylabel('Weighted Count Difference')
# plt.title('Subtracted Distributions with Weights - After Reweighting')
# plt.legend()
# plt.savefig('subtracted_distributions_with_weights_after.png')
# print('Subtracted distributions with weights saved to subtracted_distributions_with_weights_after.png')

# Plotting the distributions - after reweighting
#plt.figure(figsize=(12, 8))

# plt.hist(X1.flatten(), bins=50, weights=weights1, alpha=0.5, label='Class 0')
# plt.hist(X2.flatten(), bins=50, weights=normalized_new_weights_class_1_X2, alpha=0.5, label='Class 1')
# plt.hist(X3.flatten(), bins=50, weights=weights3, alpha=0.5, label='Class 2')
# plt.hist(X4.flatten(), bins=50, weights=normalized_new_weights_class_3_X4, alpha=0.5, label='Class 3')

# plt.xlabel('Value')
# plt.ylabel('Weighted Count')
# plt.title('Distributions of X1 - X3 and X2 - X4 with Weights')
# plt.legend()
# plt.savefig('reweighted_distributions.png')
# plt.show()

# # Plotting the distributions for class 0 and class 1 with the new weights
# plt.figure(figsize=(12, 8))

# plt.hist(X1.flatten(), bins=50, weights=weights1, alpha=0.5, label='Class 0')
# plt.hist(X2.flatten(), bins=50, weights=normalized_new_weights_class_1, alpha=0.5, label='Class 1 (Reweighted)')

# plt.xlabel('Value')
# plt.ylabel('Weighted Count')
# plt.title('Distributions for Class 0 and Class 1 with New Weights')
# plt.legend()
# plt.savefig('distributions_class_0_class_1_with_new_weights.png')

# #Reweight

# # 4. Predict the probability scores for each class - reweighting class 1 samples
# y_pred_proba2 = bst.predict(xgb.DMatrix(X2))
# class_0_proba2 = y_pred_proba2[:, 0]
# class_1_proba2 = y_pred_proba2[:, 1]
# class_2_proba2 = y_pred_proba2[:, 2]
# class_3_proba2 = y_pred_proba2[:, 3]

# y_pred_proba4 = bst.predict(xgb.DMatrix(X4))
# class_0_proba4 = y_pred_proba4[:, 0]
# class_1_proba4 = y_pred_proba4[:, 1]
# class_2_proba4 = y_pred_proba4[:, 2]
# class_3_proba4 = y_pred_proba4[:, 3]

# #Reweight
# # Derive the new weights for class 1 samples
# reweight2 = (class_0_proba2 - class_2_proba2)/ (class_1_proba2 - class_3_proba2)
# reweight4 = (class_0_proba4 - class_2_proba4)/ (class_1_proba4 - class_3_proba4)
# #reweight = class_0_proba / (class_1_proba +class_2_proba + class_3_proba)
# new_weights2 = weights2 * reweight2
# new_weights4 = weights4 * reweight4

# # Normalize new weights to sum of old weights
# sum_old_weights = np.sum(weights_X2_minus_X4)
# sum_new_weights = np.sum(new_weights2-new_weights4)
# new_weights = new_weights2-new_weights4
# normalized_new_weights = new_weights * (sum_old_weights / sum_new_weights)


# # Plotting the distributions - before reweighting
# plt.figure(figsize=(12, 8))

# plt.hist(X1_minus_X3.flatten(), bins=50, weights=weights_X1_minus_X3, alpha=0.5, label='X1 - X3')
# plt.hist(X2_minus_X4.flatten(), bins=50, weights=weights_X2_minus_X4, alpha=0.5, label='X2 - X4')

# plt.xlabel('Value')
# plt.ylabel('Weighted Count')
# plt.title('Distributions of X1 - X3 and X2 - X4 with Weights')
# plt.legend()
# plt.savefig('distributions_X1_minus_X3_X2_minus_X4_with_weights.png')
# plt.show()

# # Plotting the distributions - after reweighting
# plt.figure(figsize=(12, 8))

# plt.hist(X1_minus_X3.flatten(), bins=50, weights=weights_X1_minus_X3, alpha=0.5, label='C0 - C2')
# plt.hist(X2_minus_X4.flatten(), bins=50, weights=normalized_new_weights_class_1, alpha=0.5, label='C1 - C3')

# plt.xlabel('Value')
# plt.ylabel('Weighted Count')
# plt.title('Distributions of X1 - X3 and X2 - X4 with Weights')
# plt.legend()
# plt.savefig('reweighted_distributions_X1_minus_X3_X2_minus_X4_with_weights.png')
# plt.show()

# # Evaluate the model
# roc_auc = roc_auc_score((y_test == 0).astype(int), class_0_proba)
# print(f"ROC AUC Score for class 0: {roc_auc:.4f}")

# # Plot ROC curve
# fpr, tpr, _ = roc_curve((y_test == 0).astype(int), class_0_proba)
# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC) Curve for Class 0")
# plt.legend(loc="lower right")
# plt.savefig('testing_roc_curve_class_0.png')

# # 4. Predict the probability scores for each class in the training set
# dmatrix_X = xgb.DMatrix(X)
# y_pred_proba = bst.predict(dmatrix_X)
# class_0_proba = y_pred_proba[:, 0]
# class_1_proba = y_pred_proba[:, 1]
# class_2_proba = y_pred_proba[:, 2]
# class_3_proba = y_pred_proba[:, 3]

# # Reweight class 1 samples based on the probability of being class 0 in the training set
# class_1_indices = (y == 1)
# reweight = (class_0_proba[class_0_indices] - class_2_proba[class_1_indices]) / (class_1_proba[class_1_indices] - class_3_proba[class_1_indices])

# # Plot the reweight array
# plt.figure(figsize=(10, 6))
# plt.hist(reweight, bins=50, alpha=0.7, color='green')
# plt.xlabel('Reweight Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of Reweight Values for Class 1 Samples')
# plt.savefig('reweight_distribution.png')
# plt.show()


# # Apply reweighting to class 1 samples in the training set
# X_class_1 = X[class_1_indices]
# X_class_1_reweighted = X_class_1 * reweight[:, np.newaxis]

# print(f"Size of X_class_1: {X[class_1_indices].shape}")
# print(f"Size of reweight: {reweight.shape}")


# # Flatten the arrays for plotting
# X1_flat = X1.flatten()
# X2_flat = X2.flatten()
# X_class_1_reweighted_flat = X_class_1_reweighted.flatten()

# # Apply reweighting to X2_flat
# X2_flat_reweighted = X2_flat * reweight.repeat(n_features)

# # Print the size of the reweighted array
# print(f"Size of X2_flat: {X1_flat.shape}")  
# print(f"Size of X2_flat_reweighted: {X2_flat_reweighted.shape}")

# # Plot the distribution of class 0, class 1, and reweighted class 1
# plt.figure(figsize=(10, 6))
# plt.hist(X1_flat, bins=50, alpha=0.7, label='Class 0', color='blue')
# plt.hist(X2_flat, bins=50, alpha=0.7, label='Class 1', color='orange')
# plt.hist(X2_flat_reweighted, bins=50, alpha=0.7, label='Reweighted Class 1', color='purple')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of Class 0, Class 1, and Reweighted Class 1')
# plt.legend()
# plt.savefig('training_class_0_class_1_and_reweighted_class_1_distribution.png')