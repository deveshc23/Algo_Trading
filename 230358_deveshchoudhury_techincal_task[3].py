# -*- coding: utf-8 -*-
"""230358_DEVESHCHOUDHURY_techincal_task[3].ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1x21O5Sea-IT9iwzNfW7MIeTtE_KcV3Au
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

app_train=pd.read_csv('application_train.csv')
app_test=pd.read_csv('application_test.csv')

bureau = pd.read_csv('bureau.csv')
bureau_balance = pd.read_csv('bureau_balance.csv')
bureau.head()
app_train = app_train.dropna()
app_test = app_test.dropna()
app_test.head()

# Aggregate the bureau data
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'SK_ID_BUREAU': 'count',
    'CREDIT_ACTIVE': lambda x: (x == 'Active').sum(),
    'CREDIT_DAY_OVERDUE': 'mean',
    'AMT_CREDIT_MAX_OVERDUE': 'mean',
    'CNT_CREDIT_PROLONG': 'mean',
    'AMT_CREDIT_SUM': 'mean',
    'AMT_CREDIT_SUM_DEBT': 'mean',
    'AMT_CREDIT_SUM_LIMIT': 'mean',
    'AMT_CREDIT_SUM_OVERDUE': 'mean',
    'CREDIT_TYPE': lambda x: x.nunique()
}).reset_index()

bureau_agg.columns = ['SK_ID_CURR', 'NUM_PREV_CREDITS', 'NUM_ACTIVE_CREDITS', 'AVG_CREDIT_DAY_OVERDUE',
                      'AVG_CREDIT_MAX_OVERDUE', 'AVG_CNT_CREDIT_PROLONG', 'AVG_AMT_CREDIT_SUM',
                      'AVG_AMT_CREDIT_SUM_DEBT', 'AVG_AMT_CREDIT_SUM_LIMIT', 'AVG_AMT_CREDIT_SUM_OVERDUE',
                      'NUM_CREDIT_TYPES']

# Aggregate the bureau balance data
bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
    'MONTHS_BALANCE': 'mean',
    'STATUS': lambda x: (x == 'C').sum() / len(x)
}).reset_index()

bureau_balance_agg.columns = ['SK_ID_BUREAU', 'AVG_MONTHS_BALANCE', 'RATIO_CLOSED_STATUS']
bureau_combined = pd.merge(bureau, bureau_balance_agg, on='SK_ID_BUREAU', how='left')

bureau_combined_agg = bureau_combined.groupby('SK_ID_CURR').agg({
    'AVG_MONTHS_BALANCE': 'mean',
    'RATIO_CLOSED_STATUS': 'mean'
}).reset_index()

bureau_combined_agg.columns = ['SK_ID_CURR', 'AVG_MONTHS_BALANCE', 'AVG_RATIO_CLOSED_STATUS']
bureau_combined_agg.head()

app_train = pd.merge(app_train, bureau_agg, on='SK_ID_CURR', how='left')
app_train = pd.merge(app_train, bureau_combined_agg, on='SK_ID_CURR', how='left')

app_test = pd.merge(app_test, bureau_agg, on='SK_ID_CURR', how='left')
app_test = pd.merge(app_test, bureau_combined_agg, on='SK_ID_CURR', how='left')

app_train.head()

app_train=app_train.fillna(0)
app_test=app_test.fillna(0)
label_encoder = LabelEncoder()
for column in app_train.select_dtypes(include=['object']).columns:
    app_train[column] = label_encoder.fit_transform(app_train[column])

for column in app_test.select_dtypes(include=['object']).columns:
    app_test[column] = label_encoder.fit_transform(app_test[column])


# Separate features and target
X = app_train.drop(columns=['TARGET'])
y = app_train['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dtree = DecisionTreeClassifier()

dtree_params = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

dtree_grid = GridSearchCV(dtree, dtree_params, cv=10, scoring='accuracy')
dtree_grid.fit(X_train, y_train)
y_pred_dtree = dtree_grid.predict(X_test)
dtree_accuracy = accuracy_score(y_test, y_pred_dtree)
print(dtree_accuracy)

logreg=LogisticRegression(max_iter=1000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print(accuracy_score(y_pred,y_test))