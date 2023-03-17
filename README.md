# Car Pirce Prediction - Model Selector based on the given data

This code reads a car price dataset from a CSV file and performs <b>data cleaning and pre-processing</b> on it using the pandas library in Python. It starts by renaming columns, replacing missing values with NaN, and converting data types of some columns. The code then performs exploratory data analysis (EDA) by checking the basic statistics, shape, and unique values of the dataset. After EDA, the code removes missing values in the dataset, and fills them using mean and linear regression for some columns. Finally, the code creates a correlation matrix plot to visualize the correlation between different columns of the dataset.

This code is written in Python using Pandas, Scikit-Learn, and XGBoost libraries for machine learning. The code loads a dataset from a CSV file and performs feature selection, data scaling, and model training using several regression algorithms. The goal is to evaluate and compare the performance of different regression algorithms on the dataset and select the best one using grid search.

The dataset is loaded from the 'reg_data.csv' file using the Pandas library. Then, the correlation between the 'price' feature and each of the other features is computed using the 'corr' function of the Pandas DataFrame. This is done to identify the features that are most correlated with the target variable and could potentially have the most significant impact on the model's performance.

Next, some features are dropped from the dataset using the 'drop' function of the Pandas DataFrame. These features are chosen based on the correlation analysis performed earlier and are assumed to have little or no impact on the model's performance.

After that, the data is split into training and testing sets using the 'train_test_split' function of Scikit-Learn. The data is also scaled using the 'StandardScaler' function of Scikit-Learn to normalize the features and ensure that all features have the same scale.

Several regression algorithms are then trained on the training data, and their performance is evaluated on the testing data using the 'r2_score' and 'mean_squared_error' functions of Scikit-Learn. The algorithms used include Linear Regression, Ridge Regression, Lasso Regression, ElasticNet Regression, Decision Tree Regression, Random Forest Regression, and Gradient Boosting Regression.

The performance results of each algorithm are stored in a DataFrame, which is then sorted based on the R-squared score in descending order. The algorithm with the highest R-squared score is selected as the best algorithm.

Finally, a grid search is performed on the best algorithm to find the best hyperparameters for the model. The hyperparameters tested include alpha, l1_ratio, max_depth, n_estimators, and learning_rate, depending on the algorithm. The best hyperparameters are selected using the 'GridSearchCV' function of Scikit-Learn.
