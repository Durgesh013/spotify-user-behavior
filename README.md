# Project Overview
## Project: Spotify User Subscription Prediction

## Introduction
This project aims to predict user subscription status (Active/Inactive) for Spotify using a realistic dataset. The goal is to develop a robust classification model that can identify users at risk of churning or those likely to remain active subscribers. This predictive capability can help Spotify in targeted marketing campaigns, customer retention strategies, and optimizing service offerings.

## Dataset Description
The dataset `spotify_user_behavior_realistic_50000_rows.csv` contains various attributes related to Spotify user behavior. Key features include:
*   `user_id`: Unique identifier for each user.
*   `country`: User's country of residence.
*   `age`: User's age.
*   `signup_date`: Date when the user signed up.
*   `subscription_type`: Type of Spotify subscription (e.g., Free, Premium Individual, Premium Duo, Premium Family).
*   `subscription_status`: Target variable indicating whether the user is 'Active' or 'Inactive'.
*   `months_inactive`: Number of months the user has been inactive.
*   `inactive_3_months_flag`: Binary flag indicating if the user has been inactive for 3 or more months.
*   `ad_interaction`: User's interaction with ads (Yes/No).
*   `ad_conversion_to_subscription`: Whether ad interaction led to subscription (Yes/No).
*   `music_suggestion_rating_1_to_5`: Rating of music suggestions (1-5).
*   `avg_listening_hours_per_week`: Average listening hours per week.
*   `favorite_genre`: User's favorite music genre.
*   `most_liked_feature`: User's most liked Spotify feature.
*   `desired_future_feature`: Feature desired by the user in the future.
*   `primary_device`: Primary device used for Spotify.
*   `playlists_created`: Number of playlists created by the user.
*   `avg_skips_per_day`: Average number of song skips per day.

The target variable is `subscription_status`.

## Feature Engineering and Preprocessing
1.  **Feature Removal**: The `user_id` column was dropped as it is a unique identifier and not relevant for prediction.
2.  **Date Feature Engineering**: The `signup_date` column was transformed into numerical features:
    *   `account_age_days`: Number of days since signup until today.
    *   `signup_month`: Month of signup.
    *   `signup_year`: Year of signup.
    The original `signup_date` column was then dropped.
3.  **Data Type Categorization**:
    *   `skewed_col`: `["music_suggestion_rating_1_to_5", "months_inactive", "inactive_3_months_flag", "playlists_created", "avg_skips_per_day"]` - these were identified for power transformation.
    *   `categorical_col`: `["country", "subscription_type", 'ad_interaction', "primary_device", 'ad_conversion_to_subscription', "favorite_genre", "most_liked_feature", "desired_future_feature"]` - these were identified for one-hot encoding.
    *   `numerical_col`: `["age", "avg_listening_hours_per_week", "account_age_days", "signup_month", "signup_year"]` - these were identified for standard scaling.
4.  **Preprocessing Pipelines**:
    *   `skewed_pipe`: Applies `PowerTransformer` (Yeo-Johnson method) to handle skewed numerical features.
    *   `cat_pipe`: Applies `OneHotEncoder` to convert categorical features into numerical format.
    *   `num_pipe`: Applies `StandardScaler` to standardize numerical features.
    These individual pipelines were combined into a `ColumnTransformer` (`preprocessor`).
5.  **Train-Test Split**: The dataset was split into training and testing sets with a 80/20 ratio, using `stratify=y` to maintain the proportion of target classes.

## Modeling Approaches Explored
The following classification models were explored for predicting subscription status:

1.  **Logistic Regression**: A linear model for binary classification, often serving as a strong baseline.
    *   **Without SMOTENC**: Initial tuning performed using `GridSearchCV` on parameters like `C`, `penalty`, and `solver`.
    *   **With SMOTENC**: `SMOTENC` (Synthetic Minority Over-sampling Technique for Nominal and Continuous) was applied within a pipeline to handle potential class imbalance before applying Logistic Regression.
2.  **RandomForestClassifier**: An ensemble learning method based on decision trees, known for its robustness and accuracy. Hyperparameter tuning was performed using `RandomizedSearchCV`.
3.  **GradientBoostingClassifier**: Another powerful ensemble technique that builds trees sequentially, with each new tree correcting errors from previous ones. Hyperparameter tuning was performed using `RandomizedSearchCV`.
4.  **XGBoost (XGBClassifier)**: An optimized distributed gradient boosting library designed for speed and performance. Hyperparameter tuning was performed using `RandomizedSearchCV`.

## Best Model: Logistic Regression (without SMOTENC, with hyperparameter tuning)

After evaluating all models, the Logistic Regression model, tuned with `GridSearchCV` and without the use of `SMOTENC` (due to similar performance across SMOTENC and non-SMOTENC approaches, favoring the simpler model), was selected as the best performing model. It achieved excellent metrics on the test set.

### Hyperparameters:
The best hyperparameters for the Logistic Regression model were determined to be `C=0.01`, `penalty='l2'`, and `solver='lbfgs'`.

### Performance Metrics:
*   **Accuracy**: 0.9375
*   **Precision (Active)**: 1.00
*   **Recall (Active)**: 0.9258
*   **F1 Score (Active)**: 0.9615

### Classification Report:
```
              precision    recall  f1-score   support

      Active       1.00      0.93      0.96      8422
    Inactive       0.72      1.00      0.83      1578

    accuracy                           0.94     10000
   macro avg       0.86      0.96      0.90     10000
weighted avg       0.96      0.94      0.94     10000
```
The model shows perfect precision for the 'Active' class, meaning that when it predicts a user is active, it is always correct. The recall for the 'Inactive' class is 1.00, indicating that it correctly identifies all inactive users. However, the precision for 'Inactive' is 0.72, suggesting some active users are misclassified as inactive. Given the overall accuracy and F1-score for the 'Active' class, this model is highly effective.

### Visualizations:

#### Confusion Matrix:
![Confusion Matrix](confusion_matrix.png)
The confusion matrix shows that the model correctly predicted 7797 active users and 1578 inactive users. There were 625 active users incorrectly predicted as inactive, and 0 inactive users incorrectly predicted as active. This aligns with the high precision for 'Active' and high recall for 'Inactive'.

#### ROC Curve:
![ROC Curve](ROC-Curve.png)
The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for the best model is approximately **0.962**, indicating excellent discriminatory power between active and inactive users.

#### Precision-Recall Curve:
![Precision-Recall Curve](Precision-Recall-Curve.png)
The Precision-Recall Curve illustrates the trade-off between precision and recall for different thresholds. The curve shows high precision even at relatively high recall values, especially for the positive class.

## Model Saving and Loading
The best performing model (Logistic Regression with hyperparameter tuning, without SMOTENC) has been saved using `joblib`.

### How to Save the Model:
```python
import joblib
joblib.dump(grid, "model.pkl")
```
The `grid` object here represents the `GridSearchCV` fitted object, which internally stores the best estimator (the pipeline with the Logistic Regression model). The model was saved both locally and to Google Drive.

### How to Load the Model:
To load the saved model and make predictions on new data:
```python
import joblib

# Load the model from a local file
model = joblib.load("model.pkl")

# Or, if saved to Google Drive, load from there
# from google.colab import drive
# drive.mount('/content/drive')
# model = joblib.load("/content/drive/MyDrive/model.pkl")

# Assuming `new_data` is a pandas DataFrame with the same features as the training data
# predictions = model.predict(new_data)
# probabilities = model.predict_proba(new_data)
```

This `README.md` provides a comprehensive overview of the project, from data preparation to model selection and deployment, emphasizing the performance and utility of the chosen Logistic Regression model.
```
```
## Summary:

### Data Analysis Key Findings

*   **Project Objective**: The primary goal was to predict Spotify user subscription status (Active/Inactive) using a dataset of 50,000 user behavior records.
*   **Feature Engineering Highlights**:
    *   The `user_id` column was removed as it's not a predictive feature.
    *   The `signup_date` was transformed into `account_age_days`, `signup_month`, and `signup_year` to capture temporal patterns.
    *   Features were categorized and preprocessed using `PowerTransformer` for skewed numerical features, `OneHotEncoder` for categorical features, and `StandardScaler` for numerical features.
*   **Modeling Approaches**: Several classification models were explored, including Logistic Regression, RandomForestClassifier, GradientBoostingClassifier, and XGBoost, with hyperparameter tuning applied to each. `SMOTENC` was considered for class imbalance but ultimately not used in the best model.
*   **Best Performing Model**: A hyperparameter-tuned Logistic Regression model (without `SMOTENC`) was selected as the best performer, achieving strong results on the test set.
    *   The optimal hyperparameters were `C=0.01`, `penalty='l2'`, and `solver='lbfgs'`.
*   **Model Performance Metrics**:
    *   **Accuracy**: 0.9375
    *   **Active Class Metrics**: Precision of 1.00, Recall of 0.9258, and F1 Score of 0.9615. This indicates that when the model predicts a user is 'Active', it is always correct.
    *   **Inactive Class Metrics**: Precision of 0.72, Recall of 1.00, and F1 Score of 0.83. This means the model successfully identified all 'Inactive' users, but some 'Active' users were misclassified as 'Inactive'.
    *   **Overall Discriminatory Power**: The Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for the best model was approximately 0.962, indicating excellent ability to distinguish between classes.
*   **Confusion Matrix Observations**: The model correctly predicted 7797 'Active' users and 1578 'Inactive' users. Notably, 625 'Active' users were incorrectly predicted as 'Inactive', while zero 'Inactive' users were incorrectly predicted as 'Active'.
*   **Model Persistence**: The final best model has been saved as `model.pkl` using `joblib` for future deployment and inference.

### Insights or Next Steps

*   The model demonstrates high reliability in identifying truly active users and comprehensively capturing all inactive users, making it valuable for targeted retention campaigns.
*   Further analysis could focus on the 625 active users misclassified as inactive to understand common characteristics that might lead to these false negatives and potentially refine features or model parameters.
