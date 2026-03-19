"""
========================================================
        Spotify User Behavior ML Pipeline
========================================================

This script:
1. Loads and preprocesses data
2. Applies feature engineering
3. Builds ML pipeline
4. Performs hyperparameter tuning
5. Evaluates model
6. Saves trained model

Author: Your Name
========================================================
"""

# =========================
# 📦 IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PowerTransformer,
    FunctionTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)

# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv("spotify_user_behavior_realistic_50000_rows.csv")

# Drop unnecessary column
df.drop(columns=["user_id"], inplace=True)


# =========================
# 🛠 FEATURE ENGINEERING
# =========================
def signup_date_features(X):
    """
    Convert signup_date into useful numerical features
    """
    X = X.copy()

    X["signup_date"] = pd.to_datetime(X["signup_date"])

    X["account_age_days"] = (pd.Timestamp.today() - X["signup_date"]).dt.days
    X["signup_month"] = X["signup_date"].dt.month
    X["signup_year"] = X["signup_date"].dt.year

    # Drop original column
    X = X.drop(columns=["signup_date"])

    return X


# Transformer wrapper
date_transformer = FunctionTransformer(signup_date_features)


# =========================
# 🔢 FEATURE GROUPS
# =========================
skewed_cols = [
    "music_suggestion_rating_1_to_5",
    "months_inactive",
    "inactive_3_months_flag",
    "playlists_created",
    "avg_skips_per_day"
]

categorical_cols = [
    "country",
    "subscription_type",
    "ad_interaction",
    "primary_device",
    "ad_conversion_to_subscription",
    "favorite_genre",
    "most_liked_feature",
    "desired_future_feature"
]

numerical_cols = [
    "age",
    "avg_listening_hours_per_week",
    "account_age_days",
    "signup_month",
    "signup_year"
]


# =========================
# 🔄 PREPROCESSING PIPELINES
# =========================
skewed_pipe = Pipeline([
    ("power_transform", PowerTransformer(method="yeo-johnson"))
])

cat_pipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

num_pipe = Pipeline([
    ("scaler", StandardScaler())
])

# Combine all preprocessing
preprocessor = ColumnTransformer([
    ("skewed", skewed_pipe, skewed_cols),
    ("categorical", cat_pipe, categorical_cols),
    ("numerical", num_pipe, numerical_cols)
])


# =========================
# 🎯 TARGET & SPLIT
# =========================
X = df.drop(columns=["subscription_status"])
y = df["subscription_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=13,
    stratify=y
)


# =========================
# 🤖 MODEL + PIPELINE
# =========================
logistic = LogisticRegression()

pipe_model = Pipeline([
    ("date_features", date_transformer),
    ("preprocessing", preprocessor),
    ("logistic", logistic)
])


# =========================
# 🔍 HYPERPARAMETER TUNING
# =========================
param_grid = {
    "logistic__C": [0.01, 0.1, 1, 10],
    "logistic__penalty": ["l2"],
    "logistic__solver": ["lbfgs"]
}

grid = GridSearchCV(
    pipe_model,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)


# =========================
# 📊 MODEL EVALUATION
# =========================
y_pred = grid.predict(X_test)

print("\n========== MODEL PERFORMANCE ==========\n")
print(f"Accuracy  : {accuracy_score(y_test, y_pred)}")
print(f"Precision : {precision_score(y_test, y_pred, pos_label='Active')}")
print(f"Recall    : {recall_score(y_test, y_pred, pos_label='Active')}")
print(f"F1 Score  : {f1_score(y_test, y_pred, pos_label='Active')}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# 🔥 CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# =========================
# 📈 PRECISION-RECALL CURVE
# =========================
y_proba = grid.predict_proba(X_test)[:, 0]

precision, recall, _ = precision_recall_curve(
    y_test,
    y_proba,
    pos_label="Active"
)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("precision_recall_curve.png")
plt.show()


# =========================
# 📉 ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label="Active")
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()


# =========================
# 💾 SAVE MODEL
# =========================
joblib.dump(grid, "model.pkl")

print("\n✅ Model saved as 'model.pkl'")