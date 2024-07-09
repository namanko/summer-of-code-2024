# [Selecting a Dataset](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)
I chose a dataset from kaggle titled 'Online Payment Fraud Detection' with data like step(time of transaction), type(of transaction), amount, info about bank balance of payer and payee, and id of payer and payee.

# Data Cleaning and Preprocessing
I cleaned the data a bit, by dropping na values, checking for duplicates. I dropped the isFlaggedFraud column as I didn't find it very useful in detecting fraud transactions, isFraud seems to be enough.

# Adding Features
I added two new features, Origdiff and Destdiff, specifying the difference in balance after the transaction, I also encoded the 'type' column so that the model could see the trends of Origdiff and Destdiff for all 5 types of payments.

# Class Imbalance
There was a big problem of class imbalance as the number of non fraud transactions largely exceeded the number of fraud transactions. To avoid bias in the model, I used SMOTE to oversample the fraud transactions to 20% of non fraud transactions.

# Models and Evaluating them
I first chose the basic Logistic Regression model. The roc-auc-score was pretty nice, but the recall for non fraud transactions wasn't that great.
So, I tried XGBoost. The roc-auc-score was even better, and it had great recall for both fraud and non fraud transactions. I saved it for using it in my API.

# SHAP
This is my first time using this and I found it quite useful. It generated a simple summary plot and I got to know that the Origdiff feature had the most impact on output. (Feature Engineering works :)) ) Also running the explainer takes a long time if you run it for a large number of samples, so I simply limited it to first 1000 samples of test data.

# [API](./fraud_model_api.py)
I used FastApi to make my API using the saved model. It does a POST request on the predict endpoint with the headers.
1. amount : float
2. type_CASH_IN : int
3. type_CASH_OUT : int
4. type_DEBIT : int
5. type_PAYMENT : int
6. type_TRANSFER : int
7. Origdiff : float
8. Destdiff : float
and returns {'isFraud' : 0 or 1}


