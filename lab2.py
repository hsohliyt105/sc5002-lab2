import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score, max_error, median_absolute_error

RANDOM_STATE = 42 # creates reproducible result by fixing the random seed which makes it possible to compare and make the model better

# Hyperparameters
K = 5 # K-fold size. The train data is divided into K folds.
test_size = 0.2 # train : test = 0.8 : 0.2

n_decimals = 4
np.set_printoptions(suppress=True, precision=n_decimals) # adjust formatting for printing numpy array 

df = pd.read_csv('insurance.csv') # loads csv into raw list data
# converts raw data into usable data for training. sex and region are labelled by string, so are transformed to booleans using one-hot encoding
df['is_male'] = (df['sex'] == 'male').astype(bool)
df['is_female'] = (df['sex'] == 'female').astype(bool)
df['is_smoker'] = (df['smoker'] == 'yes').astype(bool)
region_dummies = pd.get_dummies(df['region'], prefix='region')
df = pd.concat([df, region_dummies], axis=1)

features = ['age', 
            'is_male', 
            'is_female', 
            'bmi', 
            'children', 
            'is_smoker', 
            'region_northeast', 
            'region_northwest', 
            'region_southeast', 
            'region_southwest']

X = df[features] # features are selected from the dataframe
y = df['charges']

# data are scaled (preprocessing) because each features have different ranges of values, so no single feature dominates the result
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X, y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

kf = KFold(K, shuffle=True, random_state=RANDOM_STATE) # generate folds

def print_metrics(model, X, y):
    print("--- Overall performance on test set ---")
    y_pred = model.predict(X)
    print(f"coefficients: {model.coef_}") # print the weights for the regression
    # print mse, rmse, r^2, max error to check the performance
    test_metrics = {
        'MSE': mean_squared_error(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'R²': r2_score(y, y_pred),
        'Max Error': max_error(y, y_pred)
    }
    for key, value in test_metrics.items():
        value = round(value, n_decimals)
        print(f"{key}: {value}")
    return

def cross_validate(model, X, y, cv):
    # cross validates the model with given cv method
    mse_scores = -cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=cv) # cross validate with the given kf. returns with -MSE scoring
    rmse_scores = np.sqrt(mse_scores) # calculate rmse to scale correctly
    print(f"--- Cross validation with {K} folds ---")
    print(f"Average MSE score: {np.round(mse_scores.mean(), n_decimals)}, standard deviation: {np.round(mse_scores.std(), n_decimals)}") # average mean square error score
    print(f"Average RMSE score: {np.round(rmse_scores.mean(), n_decimals)}, standard deviation: {np.round(rmse_scores.std(), n_decimals)}") # average root mean square error score
    return mse_scores

lin_reg = LinearRegression() # initialise linear regression model object
lin_reg.fit(X_train, y_train) # train the linear regression using train data
y_test_pred_lr = lin_reg.predict(X_test) # Predicts with the charges using the test data and trained model

rlr_dict = dict()
best_alpha = 0
best_scores_ridge = None
print("- Grid search for ridge regression alpha value -")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]: # different alpha values are cross validated through iteration, logarithmic scale is adopted to match any scale of data
    rlr = Ridge(alpha=alpha) # initialise ridge regression model object with the specified alpha value
    rlr.fit(X_train, y_train) # train the ridge regression using train data
    rlr_dict[alpha] = rlr
    print(f"\n-- Ridge regression with alpha {alpha} --")
    mse_scores = cross_validate(rlr, X_train, y_train, kf)

    if best_scores_ridge is None or best_scores_ridge.mean() > mse_scores.mean(): # try to get best alpha for ridge
        best_scores_ridge = mse_scores
        best_alpha = alpha

rlr = rlr_dict[best_alpha] # take the ridge regression with the best estimated alpha value
y_test_pred_rlr = rlr.predict(X_test) # Predicts with the charges using the test data and trained model

# prints metrics for each trained models
print("\n- Trained model results -")
print("\n-- Linear regression --")
cv_scores_lr = cross_validate(lin_reg, X_train, y_train, kf)
print_metrics(lin_reg, X_test, y_test)

print(f"\n-- Ridge regression with best estimated alpha value {best_alpha}--")
cv_scores_rlr = cross_validate(rlr, X_train, y_train, kf)
print_metrics(rlr, X_test, y_test)

'''
# This is to visualise the error, but i dont see the big difference even if i do so, so i just left it here for future usage or smth
fig, axes = plt.subplots(1, 2, figsize=(18,8))

ax = axes[0]
ax.scatter(y_test, y_test_pred_lr, alpha=0.5, edgecolors='k', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Charges ($)')
ax.set_ylabel('Predicted Charges ($)')
ax.set_title('Actual vs Predicted (linear regression)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(y_test, y_test_pred_rlr, alpha=0.5, edgecolors='k', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Charges ($)')
ax.set_ylabel('Predicted Charges ($)')
ax.set_title('Actual vs Predicted (ridge regression)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Test Set Evaluation: Linear vs Ridge Regression')
plt.tight_layout()
plt.show()
'''