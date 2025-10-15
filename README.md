# sc5002-lab2
Group 8

Please use the following codes to prepare the environment:

'pip3 install numpy pandas matplotlib scikit-learn seaborn'

## Data

Our dataset is chosen from kaggle and here is the link: https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset

The dataset include 7 parameters in total, which are age,bmi,sex,children,smoker,region, and we use these exactly these 7 parameters for our regression model to predict the insurance charges. Fortunately, our dataset is very clean. We did not encounter any 0s or nil in columns, which makes us easier to do the data preprocessing.

For the preprocessing, we encoded categorical variables:

sex → is_male, is_female

smoker → is_smoker

region → one-hot encoding

As a result, we have features as follows:
x0: age
x1: is_male
x2: is_female
x3: bmi
x4: children
x5: is_smoker
x6: region_northeast
x7: region_northwest
x8: region_southeast
x9: region_southwest

<img width="1193" height="649" alt="Screenshot 2025-10-15 at 3 32 26 PM" src="https://github.com/user-attachments/assets/2fe1f8f0-9bde-47f8-a5ae-f09f815fc846" />
The data heatmap describes the colinearity of each features. The features `bmi` and `region_southeast` shows mildly conlinear, where ridge model can outperform linear regression model to avoid overfitting from colinearity.

## Preprocessing

To make our results replicable, we choose 42 as the fixed random state seed.

Since the features have different range of values, the feature data had to be scaled so one particularly large value does not overly affect the overall model.

## Cross validation and hyperparameter (alpha)

<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/2ad9fcd3-d134-49d8-800f-87c9254e92be" />

In terms of the ridge regression, we adopted grid search method and k-fold for cross validation, to search the optimal ridge alpha value among tested values which are `0.01`, `0.1`, `1.0`, `10.0`, `100.0`, and we found that when alpha equals to 1 we get the optimal mse and r^2 value. The r^2 value for the optimal alpha is 0.7835 which means it can exlpan 78.35% of the train data in cross validation, which shows our model is considered good fit to the dataset. When we compare how different alpha will contribute to mse and r^2 values, we found that when alpha value equals to 100, the r^2 value shows a rapid decline and the mse value tends to have an obvious increase. This can be explained by the underfitting of the ridge regression model when the penalty value is too high. Though, these changes show the potential ability that ridge regression can do to help prevent the overfitting of the model, which is the most different part we found in our project compared to the linear regression.

## Trained model result

### Linear regression

`ŷ = 3609.149x0 + -4.6477x1 + 4.6477x2 + 2054.8851x3 + 512.4789x4 + 9544.2511x5 + 198.5836x6 + 39.8323x7 + -86.4672x8 + -148.4786x9`

`MSE: 33596915.8514, RMSE: 5796.2847, R²: 0.7836, Max Error: 22850.1365`

### Ridge regression

`ŷ = 3609.149x0 + -4.6477x1 + 4.6477x2 + 2054.8851x3 + 512.4789x4 + 9544.2511x5 + 198.5836x6 + 39.8323x7 + -86.4672x8 + -148.4786x9`

`MSE: 33604538.0379, RMSE: 5796.9421, R²: 0.7835, Max Error: 22875.9918`

## Conclusion

To compare the linear and ridge models, the trained linear regression performed slightly better than the trained ridge regression model, with MSE and R² values on test set of 33596915.8514, 0.7836, and 33604538.0379, 0.7835 for each. This means the linear model was not overfitting the train data, and use of penalty in ridge model made the prediction worse. This shows that our dataset do not have highly correlated variables or in other words outliers, and that also proves that our dataset is clean. 

Comparing the test metrics for each models, we can conclude that the linear regression can be used as the final model for the insurance charge prediction: `ŷ = 3609.149x0 + -4.6477x1 + 4.6477x2 + 2054.8851x3 + 512.4789x4 + 9544.2511x5 + 198.5836x6 + 39.8323x7 + -86.4672x8 + -148.4786x9`

Testing the model on the test set, the metrics are as follows: `MSE: 33596915.8514, RMSE: 5796.2847, R²: 0.7836, Max Error: 22850.1365`

The target data (insurance charge) shows 

Although the ordinary linear regression performed better in this dataset, in reality, we believe that dataset will not be as clean as what we done for this project. We may have more variables that are correlated to each others, more outliers which can affect the accuracy of our model, and more null data in data set; we believe in such situations, the ridge regression will take more advantages than linear regression to make a better fit prediction.

For linear regression model, data with simpler and non-colinear features could perform better over ridge model, because taking penalty term could cause underfitting, increasing the bias. For ridge regression model, data with complex, colinear feature could perform better vice versa.

The below is our printed console lines in runtime.

# Grid search for ridge regression alpha value

## Ridge regression with alpha 0.01

### Cross validation with 5 folds
Average MSE score: 37737175.5713, standard deviation: 5921826.7862
Average RMSE score: 6123.3535, standard deviation: 491.6478
Average R² score: 0.7389, standard deviation: 0.0311

## Ridge regression with alpha 0.1

### Cross validation with 5 folds
Average MSE score: 37737150.0578, standard deviation: 5921938.6763
Average RMSE score: 6123.3507, standard deviation: 491.6572
Average R² score: 0.7389, standard deviation: 0.0311

## Ridge regression with alpha 1.0

### Cross validation with 5 folds
Average MSE score: 37737033.0048, standard deviation: 5923054.7353
Average RMSE score: 6123.3336, standard deviation: 491.7504
Average R² score: 0.7389, standard deviation: 0.0311

## Ridge regression with alpha 10.0

### Cross validation with 5 folds
Average MSE score: 37749339.4412, standard deviation: 5933940.9402
Average RMSE score: 6124.273, standard deviation: 492.5645
Average R² score: 0.7388, standard deviation: 0.0311

## Ridge regression with alpha 100.0

### Cross validation with 5 folds
Average MSE score: 38941357.5903, standard deviation: 6023234.9909
Average RMSE score: 6220.9048, standard deviation: 491.6313
Average R² score: 0.7306, standard deviation: 0.0307

Best Ridge alpha = 1.0

# Trained model results

## Linear regression

### Cross validation with 5 folds
Average MSE score: 37737178.5615, standard deviation: 5921814.3507
Average RMSE score: 6123.3538, standard deviation: 491.6467
Average R² score: 0.7389, standard deviation: 0.0311

### Overall performance on test set
coefficients: [3609.149    -4.6477    4.6477 2054.8851  512.4789 9544.2511  198.5836
   39.8323  -86.4672 -148.4786]
MSE: 33596915.8514
RMSE: 5796.2847
R²: 0.7836
Max Error: 22850.1365

## Ridge regression with best estimated alpha value 1.0

### Cross validation with 5 folds
Average MSE score: 37737033.0048, standard deviation: 5923054.7353
Average RMSE score: 6123.3336, standard deviation: 491.7504
Average R² score: 0.7389, standard deviation: 0.0311

### Overall performance on test set
coefficients: [3605.5503   -4.3138    4.3138 2053.0534  512.3611 9535.0821  198.2651
   39.4642  -85.761  -148.5251]
MSE: 33604538.0379
RMSE: 5796.9421
R²: 0.7835
Max Error: 22875.9918
