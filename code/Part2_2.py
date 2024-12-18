# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load the dataset
train = pd.read_csv('application_train.csv')
print(train.shape)

# Select specific columns
selected_columns = [
    'TARGET', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
    'AMT_INCOME_TOTAL', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'OCCUPATION_TYPE', 
    'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT_W_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_3'
]
train = train[selected_columns]
print(train.shape)

# Define a function to create a missing values table
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns['Missing Values'] > 0].sort_values('% of Total Values', ascending=False).round(4)
    print("Dataframe has " + str(df.shape[1]) + " columns.\nThere are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    return mis_val_table_ren_columns

# Rename EXT_SOURCE_3 to EXT_SOURCE and display missing values
train.rename(columns={'EXT_SOURCE_3': 'EXT_SOURCE'}, inplace=True)
missing_values_table(train)

# Fill missing values for OWN_CAR_AGE and other columns
train.loc[train['FLAG_OWN_CAR'] == 'N', 'OWN_CAR_AGE'] = 0
train.loc[train['DAYS_EMPLOYED'].isnull() & train['OCCUPATION_TYPE'].isnull(), ['DAYS_EMPLOYED', 'OCCUPATION_TYPE']] = [0, 'unemployed']
train['MISSING_VALUES_COUNT'] = train.isnull().sum(axis=1)

# Display the updated missing values table
pd.set_option('display.max_rows', None)
missing_values_table(train)

# Fill missing values using mode for categorical columns and median for numerical columns
cat_cols = train.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
train[cat_cols] = imputer_cat.fit_transform(train[cat_cols])

# Label Encoding for categorical features
labelEncoder = LabelEncoder()
for col in train:
    if col != 'TARGET' and train[col].dtype == 'object':
        if len(list(train[col].unique())) <= 2:
            labelEncoder.fit(train[col])
            train[col] = labelEncoder.transform(train[col])

# One-Hot Encoding
train = pd.get_dummies(train)
print(train.shape)

# Display the missing values table after one-hot encoding
missing_values_table(train)

# Impute missing values for numerical columns using median
simple_imputer = SimpleImputer(strategy='median')
train[['OWN_CAR_AGE', 'CNT_FAM_MEMBERS']] = simple_imputer.fit_transform(train[['OWN_CAR_AGE', 'CNT_FAM_MEMBERS']])
missing_values_table(train)

# Impute missing values for numerical columns using KNN
num_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop('TARGET', errors='ignore')
imputer_knn = KNNImputer(n_neighbors=10)
train[num_cols] = imputer_knn.fit_transform(train[num_cols])

# Verify there are no missing values left
missing_values_table(train)

# Save the cleaned dataset to a CSV file
train.to_csv('train.csv', index=False)

# Load the cleaned dataset from the CSV file
train = pd.read_csv('train.csv')
print(train.shape)

# Separate features and target variable
X = train.drop(columns=['TARGET'])
y = train['TARGET']

# Calculate the correlation matrix and select features based on a threshold
correlation_matrix = X.corr()
threshold = 0.5
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > threshold)]
X_reduced = X.drop(columns=to_drop, errors='ignore')
print(X_reduced.shape)

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_reduced, y)

selected_features = X_reduced.columns[selector.get_support()]
print(selected_features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Balance the dataset using RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled_rus, y_resampled_rus = rus.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled_rus)
X_test_scaled = scaler.transform(X_test)

# Perform grid search to find the best parameters for Logistic Regression
param_grid = {
    'penalty': ['l1', 'l2'], 
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [500, 1000]
}

lr = LogisticRegression(random_state=42)
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_resampled_rus)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))

best_lr = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_lr.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')

roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC Score: {roc_auc}')