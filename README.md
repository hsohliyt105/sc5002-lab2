# sc5002-lab2
Group 8

Our dataset is chosen from kaggle and here is the link: https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset

The dataset include 7 parameters in total, which are age,bmi,sex,children,smoker,region, and we use these exactly these 7 parameters for our regression model to predict the insurance charges. Fortunately, our dataset is very clean. We did not encounter any 0s or nil in columns, which makes us easier to do the data preprocessing.

For the preprocessing we encode categorical variables:

sex → is_male, is_female

smoker → is_smoker

region → one-hot encoding

#Grid search for ridge regression alpha value


##Ridge regression with alpha 0.01


###Cross validation with 5 folds

Average MSE score: 37737175.5713, standard deviation: 5921826.7862
Average RMSE score: 6123.3535, standard deviation: 491.6478
Average R² score: 0.7389, standard deviation: 0.0311

##Ridge regression with alpha 0.1


###Cross validation with 5 folds

Average MSE score: 37737150.0578, standard deviation: 5921938.6763
Average RMSE score: 6123.3507, standard deviation: 491.6572
Average R² score: 0.7389, standard deviation: 0.0311

##Ridge regression with alpha 1.0


###Cross validation with 5 folds

Average MSE score: 37737033.0048, standard deviation: 5923054.7353
Average RMSE score: 6123.3336, standard deviation: 491.7504
Average R² score: 0.7389, standard deviation: 0.0311

##Ridge regression with alpha 10.0


###Cross validation with 5 folds

Average MSE score: 37749339.4412, standard deviation: 5933940.9402
Average RMSE score: 6124.273, standard deviation: 492.5645
Average R² score: 0.7388, standard deviation: 0.0311

##Ridge regression with alpha 100.0


###Cross validation with 5 folds

Average MSE score: 38941357.5903, standard deviation: 6023234.9909
Average RMSE score: 6220.9048, standard deviation: 491.6313
Average R² score: 0.7306, standard deviation: 0.0307

Best Ridge alpha = 1.0

#Trained model results


##Linear regression


###Cross validation with 5 folds

Average MSE score: 37737178.5615, standard deviation: 5921814.3507
Average RMSE score: 6123.3538, standard deviation: 491.6467
Average R² score: 0.7389, standard deviation: 0.0311

###Overall performance on test set

coefficients: [3609.149    -4.6477    4.6477 2054.8851  512.4789 9544.2511  198.5836
   39.8323  -86.4672 -148.4786]
MSE: 33596915.8514
RMSE: 5796.2847
R²: 0.7836
Max Error: 22850.1365

##Ridge regression with best estimated alpha value 1.0


###Cross validation with 5 folds

Average MSE score: 37737033.0048, standard deviation: 5923054.7353
Average RMSE score: 6123.3336, standard deviation: 491.7504
Average R² score: 0.7389, standard deviation: 0.0311

###Overall performance on test set

coefficients: [3605.5503   -4.3138    4.3138 2053.0534  512.3611 9535.0821  198.2651
   39.4642  -85.761  -148.5251]
MSE: 33604538.0379
RMSE: 5796.9421
R²: 0.7835
Max Error: 22875.9918

