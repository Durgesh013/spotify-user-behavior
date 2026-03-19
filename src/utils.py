import pandas as pd

def signup_date_features(X):
    X = X.copy()
    X["signup_date"] = pd.to_datetime(X["signup_date"])

    X["account_age_days"] = (pd.Timestamp.today() - X["signup_date"]).dt.days
    X["signup_month"] = X["signup_date"].dt.month
    X["signup_year"] = X["signup_date"].dt.year

    return X.drop(columns=["signup_date"])