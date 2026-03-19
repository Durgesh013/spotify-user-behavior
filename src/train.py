"""
========================================================
                TRAINING SCRIPT
========================================================
- Loads dataset
- Applies preprocessing + feature engineering
- Trains model using GridSearchCV
- Saves trained pipeline to /models/model.pkl
========================================================
"""

# =========================
# 📦 IMPORTS
# =========================
import os
import pandas as pd
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

from utils import signup_date_features


# =========================
# 📁 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "spotify_user_behavior_realistic_50000_rows.csv")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")


# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv(data_path)
df.drop(columns=["user_id"], inplace=True)


# =========================
# 🎯 TARGET & FEATURES
# =========================
X = df.drop(columns=["subscription_status"])
y = df["subscription_status"]


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
# 🔄 PREPROCESSING
# =========================
preprocessor = ColumnTransformer([
    ("skew", PowerTransformer(method="yeo-johnson"), skewed_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])


# =========================
# 🤖 PIPELINE
# =========================
pipeline = Pipeline([
    ("date_features", FunctionTransformer(signup_date_features)),
    ("preprocessing", preprocessor),
    ("model", LogisticRegression())
])


# =========================
# 🔍 HYPERPARAMETERS
# =========================
param_grid = {
    "model__C": [0.01, 0.1, 1, 10]
}


# =========================
# 📊 TRAIN
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=13,
    stratify=y
)

from sklearn.metrics import make_scorer, f1_score

f1_scorer = make_scorer(f1_score, pos_label="Active")

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring=f1_scorer,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)


# =========================
# 💾 SAVE MODEL
# =========================
joblib.dump(grid, model_path)

print(f"\n✅ Model saved at: {model_path}")