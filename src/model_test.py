"""
========================================================
                MODEL INFERENCE SCRIPT
========================================================
- Loads trained model
- Takes sample input
- Returns prediction + confidence
========================================================
"""

# =========================
# 📦 IMPORTS
# =========================
import os
import joblib
import pandas as pd

from utils import signup_date_features  # REQUIRED for pipeline


# =========================
# 📁 PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")


# =========================
# 📂 LOAD MODEL
# =========================
model = joblib.load(model_path)


# =========================
# 🔮 PREDICTION FUNCTION
# =========================
def predict(data: dict):
    """
    Takes input as dictionary
    Returns prediction + confidence
    """

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df).max()

    return {
        "prediction": prediction,
        "confidence": float(probability)
    }


# =========================
# 🧪 TEST RUN
# =========================
if __name__ == "__main__":

    sample_input = {
        "age": 25,
        "country": "India",
        "subscription_type": "Free",
        "ad_interaction": "High",
        "primary_device": "Mobile",
        "ad_conversion_to_subscription": "No",
        "favorite_genre": "Pop",
        "most_liked_feature": "Recommendations",
        "desired_future_feature": "Better UI",
        "music_suggestion_rating_1_to_5": 4,
        "months_inactive": 1,
        "inactive_3_months_flag": 0,
        "playlists_created": 10,
        "avg_skips_per_day": 5,
        "avg_listening_hours_per_week": 15,
        "signup_date": "2022-01-01"
    }

    result = predict(sample_input)

    print("\n🔮 Prediction Result:")
    print(result)