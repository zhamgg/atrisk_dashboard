#! /usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

csv_path = 'data/company_data.csv' 

df = pd.read_csv(csv_path)

terminated_mask = df['CompanyStatus'].isin(['Terminated', 'InActive'])

terminated_companies = df[terminated_mask]
terminated_companies = terminated_companies.drop_duplicates(subset=['CompanyEntityId'], keep='first')

df = pd.concat([
    df[~terminated_mask],
    terminated_companies
])

# Filter out demo and InActive companies
df = df[~df['CompanyStatus'].str.contains('demo', case=False, na=False)]
df = df[df['CompanyStatus'] != 'InActive']

base_features = [
    "Fiduciary Investment Review", "PlanFees Prism", "Year", "ProducerGroup", 
    "Provider Analysis", "Prism365", "TDF Analyzer", "RFP Express", "Fiduciary Plan Review", 
    "Region"
]

optimized_features = base_features

df.loc[:, 'target'] = (df['CompanyStatus'].isin(['Terminated', 'InActive'])).astype(int)


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

categorical_cols = [col for col in categorical_cols if col not in ['CompanyName', 'CompanyAlias', 'CompanyEntityId', 'CompanyStatus']]

for col in categorical_cols:
    df[col] = df[col].astype(str)
    df[col] = LabelEncoder().fit_transform(df[col])

company_info = df[['CompanyName', 'CompanyAlias', 'CompanyEntityId', 'CompanyStatus']].copy()

df = df.drop(columns=['CompanyName', 'CompanyAlias', 'CompanyEntityId'])

X_full = df[base_features]
X = df[optimized_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=imbalance_ratio,
    eval_metric='logloss',
    random_state=42,
    colsample_bytree=0.7,  
    gamma=0.1,  
    max_depth=4,
    min_child_weight=2, 
    n_estimators=150,  
    subsample=0.8,  
    learning_rate=0.1, 
)

model.fit(X_train, y_train)

y_full_pred_proba = model.predict_proba(X_full)[:, 1]

results_df = pd.DataFrame({
    'CompanyName': company_info['CompanyName'],
    'CompanyAlias': company_info['CompanyAlias'],
    'CompanyEntityId': company_info['CompanyEntityId'],
    'Termination_Probability': y_full_pred_proba,
    'Risk_Level': pd.cut(y_full_pred_proba, 
                        bins=[0, 0.2, 0.4, 0.6, 1.0],  # More conservative thresholds
                        labels=['Low', 'Medium', 'High', 'Very High'],
                        include_lowest=True),
    'Actual_Status': company_info['CompanyStatus'],
    'Prediction_Correct': (y_full_pred_proba >= 0.5) == (company_info['CompanyStatus'] == 'Terminated')
})

# Sort by probability in descending order
results_df = results_df.sort_values('Termination_Probability', ascending=False)

# Save to CSV
results_df.to_csv('data/demo_predictions.csv', index=False)
