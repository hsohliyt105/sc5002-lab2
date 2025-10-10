import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score, max_error, median_absolute_error

RANDOM_STATE = 42 # creates reproducible result by fixing the random seed which makes it possible to compare and make the model better

K = 5 # K-fold size. The train data is divided into K folds.
test_size = 0.2 # train : test = 0.8 : 0.2

raw_data = list(csv.reader(open('insurance 2.csv')))[1:] # loads csv into raw list data
X = [] # features list declaration
y = [] # target label list declaration

# converts raw data into usable data for training. sex and region are labelled by string, so are transformed to booleans using one-hot encoding
for row in raw_data:
    X.append(np.array([ # features
        int(row[0]), #age
        row[1]=="male", #is_male
        row[1]=="femalt e", #is_female
        float(row[2]), #bmi
        int(row[3]), #children
        row[4]=="yes", #is_smoker
        row[5]=="northeast", #is_northeast
        row[5]=="northwest", #is_northwest
        row[5]=="southeast", #is_southeast
        row[5]=="southwest", #is_southwest     
    ]))
    y.append(float(row[6])) # target label

X = np.array(X) # list is converted into numpy array for ease of data manipulation
y = np.array(y)

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X, y) # data are scaled (preprocessing) because each features have different ranges of values, so no single feature dominates the result

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

'''
def cross_validation(model, X, y):
    # cross validation function. validates the model using k-fold validation.
    kf = KFold(K, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf)
    rmse_scores = np.sqrt(-scores)
    print("Scores:", rmse_scores) # root mean square error score
    print("Mean:", rmse_scores.mean()) 
    print("StandardDeviation:", rmse_scores.std())
'''

lin_reg = LinearRegression() # initialise linear regression model object
lin_reg.fit(X_train, y_train) # train the linear regression using train data
y_test_pred_lr = lin_reg.predict(X_test) # test the trained regression using test data
#cross_validation(lin_reg, X_train, y_train) # cross validate using k-fold

print(f"linear regression coefficients: {lin_reg.coef_}") # print the weights for the regression
test_metrics_lr = {
    'MSE': mean_squared_error(y_test, y_test_pred_lr),
    'R²': r2_score(y_test, y_test_pred_lr),
    'Max Error': max_error(y_test, y_test_pred_lr)
}
print(test_metrics_lr) # print mse, r^2, max error to check the performance

for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]: # different alpha values are tested through iteration
    rlr = Ridge(alpha=alpha) # initialise ridge regression model object with the specified alpha value
    #cross_validation(rlr, X_train, y_train) # cross validate using k-fold
    rlr.fit(X_train, y_train) # train the ridge regression using train data
    y_test_pred_rlr= rlr.predict(X_test) # test the trained regression using test data

    print(f"linear regression coefficients: {lin_reg.coef_}") # print the weights for the regression
    test_metrics_rlr = {
        'MSE': mean_squared_error(y_test, y_test_pred_rlr),
        'R²': r2_score(y_test, y_test_pred_rlr),
        'Max Error': max_error(y_test, y_test_pred_rlr)
    }
    print(test_metrics_rlr) # print mse, r^2, max error to check the performance

'''
This is to visualise the error, but i dont see the big difference even if i do so, so i just left it here for future usage or smth
fig, axes = plt.subplots(1, 2, figsize=(18,12))

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

plt.suptitle(f'Test Set Evaluation: linear vs. ridge')
plt.tight_layout()
plt.show()
'''
