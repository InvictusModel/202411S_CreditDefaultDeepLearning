import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Set the working directory
import os
os.chdir('C:/User/xxx')

# Import data
data = pd.read_csv('application_train.csv')

# Select variables
data['na_count'] = data.isnull().sum(axis=1)
data = data[['SK_ID_CURR', 'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OWN_CAR_AGE', 'FLAG_OWN_CAR', 'CNT_FAM_MEMBERS', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'NAME_EDUC_TYPE', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE']]

# Deal with missing values
median_EXT_SOURCE_3 = data['EXT_SOURCE_3'].median()
data['EXT_SOURCE'] = data['EXT_SOURCE_3'].fillna(median_EXT_SOURCE_3)
data.drop('EXT_SOURCE_3', axis=1, inplace=True)

data.loc[data['FLAG_OWN_CAR'] == 'N', 'OWN_CAR_AGE'] = 0
median_OWN_CAR_AGE = data['OWN_CAR_AGE'].median()
data['OWN_CAR_AGE'] = data['OWN_CAR_AGE'].fillna(median_OWN_CAR_AGE)

median_CNT_FAM_MEMBERS = data['CNT_FAM_MEMBERS'].median()
data['CNT_FAM_MEMBERS'] = data['CNT_FAM_MEMBERS'].fillna(median_CNT_FAM_MEMBERS)

# Check for missing values
print(data.isnull().sum())

# Divide the original sample into training and test sets
target0 = data[data['TARGET'] == 0]
target1 = data[data['TARGET'] == 1]
s = np.random.choice(target0.index, size=len(target1), replace=False)
target0 = target0.loc[s]

data1 = pd.concat([target0, target1], axis=0)
trainset, testset = train_test_split(data1, test_size=0.3, random_state=1030)

# Logistic regression model
logit_model = LogisticRegression()
logit_model.fit(trainset.drop('SK_ID_CURR', axis=1), trainset['TARGET'])
logit_prob = logit_model.predict_proba(testset.drop('SK_ID_CURR', axis=1))[:, 1]
logit_predict = (logit_prob > 0.5).astype(int)
logit_confusion = confusion_matrix(testset['TARGET'], logit_predict)
logit_accuracy = np.round((logit_confusion[0, 0] + logit_confusion[1, 1]) / (logit_confusion[0, 0] + logit_confusion[0, 1] + logit_confusion[1, 0] + logit_confusion[1, 1]), 4)
logit_sensitivity = np.round(logit_confusion[1, 1] / (logit_confusion[1, 0] + logit_confusion[1, 1]), 4)
logit_specificity = np.round(logit_confusion[0, 0] / (logit_confusion[0, 0] + logit_confusion[0, 1]), 4)

# Random Forest machine learning model
rf_model = RandomForestClassifier()
rf_model.fit(trainset.drop('SK_ID_CURR', axis=1), trainset['TARGET'])
rf_predict = rf_model.predict(testset.drop('SK_ID_CURR', axis=1))
rf_prob = rf_model.predict_proba(testset.drop('SK_ID_CURR', axis=1))[:, 1]
rf_confusion = confusion_matrix(testset['TARGET'], rf_predict)
rf_accuracy = np.round((rf_confusion[0, 0] + rf_confusion[1, 1]) / (rf_confusion[0, 0] + rf_confusion[0, 1] + rf_confusion[1, 0] + rf_confusion[1, 1]), 4)
rf_sensitivity = np.round(rf_confusion[1, 1] / (rf_confusion[1, 0] + rf_confusion[1, 1]), 4)
rf_specificity = np.round(rf_confusion[0, 0] / (rf_confusion[0, 0] + rf_confusion[0, 1]), 4)

# Evaluate the prediction performance
fpr_logit, tpr_logit, _ = roc_curve(testset['TARGET'], logit_prob)
fpr_rf, tpr_rf, _ = roc_curve(testset['TARGET'], rf_prob)
plt.figure()
plt.plot(fpr_logit, tpr_logit, color='#e7bcbc', lw=2, label='Logit regression model')
plt.plot(fpr_rf, tpr_rf, color='#8d1f17', lw=2, label='Random forest machine learning model')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Interpret the effects of variables on prediction
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot Gini importance
plt.figure()
plt.title('Feature Importances (Gini index)')
plt.bar(range(trainset.drop('SK_ID_CURR', axis=1).shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(trainset.drop('SK_ID_CURR', axis=1).shape[1]), trainset.drop('SK_ID_CURR', axis=1).columns[indices], rotation=90)
plt.xlim([-1, trainset.drop('SK_ID_CURR', axis=1).shape[1]])
plt.show()

# Plot Accuracy importance
plt.figure()
plt.title('Feature Importances (Accuracy)')
plt.bar(range(trainset.drop('SK_ID_CURR', axis=1).shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(trainset.drop('SK_ID_CURR', axis=1).shape[1]), trainset.drop('SK_ID_CURR', axis=1).columns[indices], rotation=90)
plt.xlim([-1, trainset.drop('SK_ID_CURR', axis=1).shape[1]])
plt.show()