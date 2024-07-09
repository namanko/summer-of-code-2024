# Selecting a dataset
I chose a dataset from kaggle titled 'Online Payment Fraud Detection' with data like step(time of transaction), type(of transaction), amount, info about bank balance of payer and payee, and id of payer and payee.

# Data cleaning and preprocessing
I cleaned the data a bit, by dropping na values, checking for duplicates. I dropped the isFlaggedFraud column as I didn't find it very useful in detecting fraud transactions, isFraud seems to be enough.

# Adding Features
I added two new features, Origdiff and Destdiff, specifying the difference in balance after the transaction, I also encoded the 'type' column so that the model could see the trends of Origdiff and Destdiff for all 5 types of payments.

# Class imbalance
There was a big problem of class imbalance as the number of non fraud transactions largely exceeded the number of fraud transactions. To avoid bias in the model, I used SMOTE to oversample the fraud transactions to 20% of non fraud transactions.

# Models and Evaluating them
I first chose the basic Logistic Regression model. The roc-auc-score was pretty nice, but the recall for non fraud transactions, wasn't that great.
So, I tried XGBoost. the roc-auc-score was even better, and it had great recall for both fraud and non fraud transactions. I saved it for using it in my API.

# SHAP
This is my first time using this and I found it quite useful. It generated a simple summary plot and I got to know that the Origdiff feature had the most impact on output. (Feature Engineering works :)) ) Also running the explainer takes a long time if you run it for a large number of samples, so I simply limited it to first 1000 samples of test data.

# API
I used FastApi to make my API using the saved model. It does a POST request on the predict endpoint with the headers 
    - amount : float
    - type_CASH_IN : int
    - type_CASH_OUT : int
    - type_DEBIT : int
    - type_PAYMENT : int
    - type_TRANSFER : int
    - Origdiff : float
    - Destdiff : float
and returns {'isFraud' : 0 or 1}


