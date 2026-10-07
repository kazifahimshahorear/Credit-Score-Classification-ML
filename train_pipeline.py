import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score

print("=== 1. DATA PREPROCESSING & ERROR HANDLING ===")
# Load dataset
train = pd.read_csv('train.csv', low_memory=False)
test = pd.read_csv('test.csv', low_memory=False)

target = 'Credit_Score'

# Drop irrelevant columns
drop_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
train_clean = train.drop(columns=drop_cols, errors='ignore')
test_ids = test['ID'] if 'ID' in test.columns else None
test_clean = test.drop(columns=drop_cols, errors='ignore')

# Function for noise & symbol cleaning
def clean_numeric(val):
    if pd.isna(val): 
        return np.nan
    val = str(val).replace('_', '').replace('-', '')
    try:
        return float(val)
    except:
        return np.nan

numeric_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
                'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']

for col in numeric_cols:
    if col in train_clean.columns: train_clean[col] = train_clean[col].apply(clean_numeric)
    if col in test_clean.columns: test_clean[col] = test_clean[col].apply(clean_numeric)

# Imputation & Categorical Label Encoding
encoders = {}
for col in train_clean.columns:
    if col != target:
        if train_clean[col].dtype == 'object' or test_clean[col].dtype == 'object':
            train_clean[col] = train_clean[col].astype(str).fillna('Unknown')
            test_clean[col] = test_clean[col].astype(str).fillna('Unknown')
            
            le = LabelEncoder()
            le.fit(list(train_clean[col].unique()) + list(test_clean[col].unique()))
            train_clean[col] = le.transform(train_clean[col])
            test_clean[col] = le.transform(test_clean[col])
            encoders[col] = le
        else:
            median_val = train_clean[col].median()
            train_clean[col] = train_clean[col].fillna(median_val)
            test_clean[col] = test_clean[col].fillna(median_val)

print("Data Preprocessing Completed successfully.")

print("\n=== 2. DATA VISUALIZATION ===")
# Target distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=train_clean, x=target)
plt.title('Credit Score Distribution')
plt.savefig('credit_score_distribution.png')
plt.close()

# Feature Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train_clean.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
print("Plots saved as 'credit_score_distribution.png' & 'correlation_heatmap.png'.")

print("\n=== 3. FEATURE SELECTION ===")
X = train_clean.drop(columns=[target])
y = train_clean[target]

# Selecting top 15 features using ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Selected Top Features:", list(selected_features))

X_final = X[selected_features]
X_test_final = test_clean[selected_features]

print("\n=== 4. CROSS VALIDATION & HYPERPARAMETER OPTIMIZATION ===")
X_train, X_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# Grid Search CV setup
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Validation performance
y_pred = best_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

print("\n=== 5. MODEL SAVING & SUBMISSION PREDICTION ===")
# Save best model and selected feature list
joblib.dump(best_model, 'best_credit_score_model.pkl')
joblib.dump(list(selected_features), 'selected_features.pkl')
print("Model saved to 'best_credit_score_model.pkl'.")

# Final predictions for test set
test_preds = best_model.predict(X_test_final)
submission = pd.DataFrame({'ID': test_ids, 'Credit_Score': test_preds})
submission.to_csv('submission_final.csv', index=False)
print("Submission saved to 'submission_final.csv'.")