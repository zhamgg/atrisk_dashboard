#! /usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib
import logging

# Configure pandas to opt-in to future behavior for downcasting
pd.set_option('future.no_silent_downcasting', True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define expected headers
EXPECTED_HEADERS = [
    'ProducerGroup', 'CompanyEntityId', 'CompanyName', 'CompanyAlias', 'CompanyStatus',
    'Region', 'Year', 'Month', 'Fiduciary Investment Review', 'Fiduciary Plan Review',
    'Provider Analysis', 'Provider Overview', 'PLANavigator', 'HSA Investment Review',
    'RFP Express', 'PlanFees Prism', 'Prism365', 'Stable Value Analyzer', 'TDF Analyzer',
    'Rollover Ready'
]

# preprocessing
csv_path = 'data/company_data.csv' 

df = pd.read_csv(csv_path)

# Check if number of columns match
if len(df.columns) != len(EXPECTED_HEADERS):
    raise ValueError(f"CSV has {len(df.columns)} columns but expected {len(EXPECTED_HEADERS)} columns")

# Replace headers with expected ones
df.columns = EXPECTED_HEADERS

logging.info(f"Initial number of companies: {len(df)}")

terminated_mask = df['CompanyStatus'].isin(['Terminated', 'InActive'])

terminated_companies = df[terminated_mask]
terminated_companies = terminated_companies.drop_duplicates(subset=['CompanyEntityId'], keep='first')

df = pd.concat([
    df[~terminated_mask],
    terminated_companies
])

logging.info(f"Number of companies after removing duplicates: {len(df)}")
logging.info(f"Number of duplicate terminated companies removed: {len(pd.read_csv(csv_path)) - len(df)}")

# Filter out demo companies 
df = df[~df['CompanyStatus'].str.contains('demo', case=False, na=False)]

# Split data into pre-current year and post-current year
pre_cutoff_mask = df['Year'] < 2025
post_cutoff_mask = df['Year'] >= 2025

pre_cutoff_df = df[pre_cutoff_mask]
post_cutoff_df = df[post_cutoff_mask]

logging.info(f"\nNumber of companies pre-cutoff: {len(pre_cutoff_df)}")
logging.info(f"Number of companies post-cutoff: {len(post_cutoff_df)}")

optimized_features = [
    "Fiduciary Investment Review", "PlanFees Prism", "Year", "ProducerGroup", 
    "Provider Analysis", "Prism365", "TDF Analyzer", "RFP Express", "Fiduciary Plan Review", 
    "Region"
]

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
pre_cutoff_df = pre_cutoff_df.copy()
post_cutoff_df = post_cutoff_df.copy()

# Set target using proper indexing
pre_cutoff_df.loc[:, 'target'] = (pre_cutoff_df['CompanyStatus'].isin(['Terminated', 'InActive'])).astype(int)

categorical_cols = pre_cutoff_df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ['CompanyName', 'CompanyAlias', 'CompanyEntityId', 'CompanyStatus']]

label_encoders = {}
for col in categorical_cols:
    pre_cutoff_df.loc[:, col] = pre_cutoff_df[col].astype(str)
    post_cutoff_df.loc[:, col] = post_cutoff_df[col].astype(str)
    
    unique_values = pre_cutoff_df[col].unique()
    
    label_encoders[col] = LabelEncoder()
    label_encoders[col].fit(unique_values)
    
    pre_cutoff_df.loc[:, col] = label_encoders[col].transform(pre_cutoff_df[col])
    
    post_cutoff_df.loc[:, col] = post_cutoff_df[col].apply(
        lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ 
        else len(label_encoders[col].classes_)  # Assign to "Unknown" category
    )

for col in optimized_features:
    pre_cutoff_df.loc[:, col] = pd.to_numeric(pre_cutoff_df[col], errors='coerce')
    post_cutoff_df.loc[:, col] = pd.to_numeric(post_cutoff_df[col], errors='coerce')

# Handle fillna with explicit downcasting
pre_cutoff_df = pre_cutoff_df.fillna(0).infer_objects(copy=False)
post_cutoff_df = post_cutoff_df.fillna(0).infer_objects(copy=False)

pre_cutoff_company_info = pre_cutoff_df[['CompanyName', 'CompanyAlias', 'CompanyEntityId', 'CompanyStatus']].copy()
post_cutoff_company_info = post_cutoff_df[['CompanyName', 'CompanyAlias', 'CompanyEntityId', 'CompanyStatus']].copy()

pre_cutoff_df = pre_cutoff_df.drop(columns=['CompanyName', 'CompanyAlias', 'CompanyEntityId'])
post_cutoff_df = post_cutoff_df.drop(columns=['CompanyName', 'CompanyAlias', 'CompanyEntityId'])

X_pre_cutoff = pre_cutoff_df[optimized_features]
y_pre_cutoff = pre_cutoff_df['target']

X_post_cutoff = post_cutoff_df[optimized_features]

# model building using pre-cutoff data

X_train, X_test, y_train, y_test = train_test_split(X_pre_cutoff, y_pre_cutoff, test_size=0.2, random_state=42, stratify=y_pre_cutoff)

imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=imbalance_ratio, 
    eval_metric='logloss',
    random_state=42,
    colsample_bytree=0.6, 
    max_depth=6,  
    min_child_weight=1,
    n_estimators=200,  
    subsample=0.9,  
    learning_rate=0.03,  
    reg_alpha=0.05,  
    reg_lambda=0.5,  
)

model.fit(X_train, y_train)

# model testing using post-2025 data
post_cutoff_pred_proba = model.predict_proba(X_post_cutoff)[:, 1]

# store results in a specific format for the app
post_cutoff_results = pd.DataFrame({
    'CompanyName': post_cutoff_company_info['CompanyName'],
    'CompanyAlias': post_cutoff_company_info['CompanyAlias'],
    'CompanyEntityId': post_cutoff_company_info['CompanyEntityId'],
    'Termination_Probability': post_cutoff_pred_proba,
    'Risk_Level': pd.cut(post_cutoff_pred_proba, 
                        bins=[0, 0.2, 0.4, 0.6, 1.0],
                        labels=['Low', 'Medium', 'High', 'Very High'],
                        include_lowest=True),
    'Actual_Status': post_cutoff_company_info['CompanyStatus']
})

post_cutoff_results = post_cutoff_results.sort_values('Termination_Probability', ascending=False)

post_cutoff_results.to_csv('data/terminations.csv', index=False)

y_pred_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.25  
y_pred = (y_pred_proba >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
auc = roc_auc_score(y_test, y_pred_proba)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = 2 * (precision * recall) / (precision + recall)

logging.info("\nTest Set Performance (pre-cutoff data):")
logging.info(f"AUC: {auc}, Recall: {recall}, Precision: {precision}, F1: {f1}")
logging.info(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")

# Save the model
joblib.dump(model, 'data/xgb_model.joblib')

# X_test.to_csv('X_test.csv', index=False)
#pd.DataFrame(y_test, columns=['target']).to_csv('y_test.csv', index=False)



