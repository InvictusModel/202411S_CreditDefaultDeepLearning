"""
ID: Unique identifier for each entry in the dataset.
Customer_ID: Identifier for each customer.
Month: Month of data collection.
Name: Name of the customer.
Age: Age of the customer.
SSN: Social Security Number of the customer.
Occupation: Occupation of the customer.
Annual_Income: Annual income of the customer.
Monthly_Inhand_Salary: Monthly salary after deductions.
Num_Bank_Accounts: Number of bank accounts the customer has.
Num_Credit_Card: Number of credit cards the customer has.
Interest_Rate: Interest rate applied on loans.
Num_of_Loan: Number of loans the customer has.
Type_of_Loan: Type of loan taken by the customer.
Delay_from_due_date: Number of days delayed from due date for payments.
Num_of_Delayed_Payment: Number of delayed payments made by the customer.
Changed_Credit_Limit: Indicates if the credit limit has been changed.
Num_Credit_Inquiries: Number of credit inquiries made by the customer.
Credit_Mix: Mix of different types of credit accounts held by the customer.
Outstanding_Debt: Amount of outstanding debt.
Credit_Utilization_Ratio: Ratio of credit used to credit available.
Credit_History_Age: Age of credit history.
Payment_of_Min_Amount: Indicates if minimum payment amount is met.
Total_EMI_per_month: Total Equated Monthly Installment (EMI) paid by the customer.
Amount_invested_monthly: Amount invested monthly by the customer.
Payment_Behaviour: Payment behavior of the customer.
Monthly_Balance: Monthly balance in the account.
Credit_Score: Target variable - credit score of the customer.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")


def knn_impute_column(df, column, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[[column]] = imputer.fit_transform(df[[column]])
    return df


def parse_years_and_months_to_months(age):
    if isinstance(age, str):
        age_parts = age.split(' Years and ')
        years = int(age_parts[0]) if 'Years' in age else 0
        months_str = age_parts[1].split(' Months')[0] if 'Months' in age_parts[1] else '0'
        months = int(months_str)
        total_months = years * 12 + months
        return total_months
    else:
        return 0


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df["Age"] = df["Age"].str.replace("-", "").str.replace("_", "").astype("int")
df.drop(df[df["Age"] > 100].index, inplace=True)
df_test["Age"] = df_test["Age"].str.replace("-", "").str.replace("_", "").astype("int")
df_test.drop(df_test[df_test["Age"] > 100].index, inplace=True)

df["Occupation"] = df["Occupation"].str.replace("_______", "Other")
df_test["Occupation"] = df_test["Occupation"].str.replace("_______", "Other")

df["Annual_Income"] = df["Annual_Income"].str.replace("_", "").astype("float")
df_test["Annual_Income"] = df_test["Annual_Income"].str.replace("_", "").astype("float")

df["Monthly_Inhand_Salary"] = df["Monthly_Inhand_Salary"].fillna(df["Annual_Income"]/12)
df_test["Monthly_Inhand_Salary"] = df_test["Monthly_Inhand_Salary"].fillna(df_test["Annual_Income"]/12)

df["Num_of_Loan"] = df["Num_of_Loan"].str.replace("_", "").str.replace("-", "").astype("int")
df_test["Num_of_Loan"] = df_test["Num_of_Loan"].str.replace("_", "").str.replace("-", "").astype("int")

df["Type_of_Loan"] = df["Type_of_Loan"].fillna("Unknown")
df_test["Type_of_Loan"] = df_test["Type_of_Loan"].fillna("Unknown")

df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].str.replace("_", "").str.replace("-", "")
df_test["Num_of_Delayed_Payment"] = df_test["Num_of_Delayed_Payment"].str.replace("_", "").str.replace("-", "")
knn_impute_column(df, 'Num_of_Delayed_Payment')
knn_impute_column(df_test, 'Num_of_Delayed_Payment')

df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].str.replace("_", "").str.replace("-", "")
df['Changed_Credit_Limit'] = pd.to_numeric(df['Changed_Credit_Limit'], errors='coerce')
mean_value = df['Changed_Credit_Limit'].mean()
df['Changed_Credit_Limit'].fillna(mean_value, inplace=True)
df_test["Changed_Credit_Limit"] = df_test["Changed_Credit_Limit"].str.replace("_", "").str.replace("-", "")
df_test['Changed_Credit_Limit'] = pd.to_numeric(df_test['Changed_Credit_Limit'], errors='coerce')
mean_value = df_test['Changed_Credit_Limit'].mean()
df_test['Changed_Credit_Limit'].fillna(mean_value, inplace=True)

knn_impute_column(df, 'Num_Credit_Inquiries')
knn_impute_column(df_test, 'Num_Credit_Inquiries')

df["Credit_Mix"] = df["Credit_Mix"].str.replace("_", "Unknown")
df_test["Credit_Mix"] = df_test["Credit_Mix"].str.replace("_", "Unknown")

df["Outstanding_Debt"] = df["Outstanding_Debt"].str.replace("_", "").astype("float")
df_test["Outstanding_Debt"] = df_test["Outstanding_Debt"].str.replace("_", "").astype("float")

df['Credit_History_Age_Months'] = df['Credit_History_Age'].apply(parse_years_and_months_to_months)
df.drop(columns=['Credit_History_Age'], inplace=True)
df_test['Credit_History_Age_Months'] = df_test['Credit_History_Age'].apply(parse_years_and_months_to_months)
df_test.drop(columns=['Credit_History_Age'], inplace=True)

df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].apply(lambda x: 'No' if x == 'NM' else x)
df_test['Payment_of_Min_Amount'] = df_test['Payment_of_Min_Amount'].apply(lambda x: 'No' if x == 'NM' else x)

df['Amount_invested_monthly'] = df['Amount_invested_monthly'].replace('__10000__', np.nan)

df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly']\
    .transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
df["Amount_invested_monthly"] = df["Amount_invested_monthly"].astype(float)
df_test['Amount_invested_monthly'] = df_test['Amount_invested_monthly'].replace('__10000__', np.nan)
df_test['Amount_invested_monthly'] = df_test.groupby('Customer_ID')['Amount_invested_monthly']\
    .transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
df_test["Amount_invested_monthly"] = df_test["Amount_invested_monthly"].astype(float)

df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)
df['Payment_Behaviour'] = df['Payment_Behaviour'].fillna(method='ffill')
df_test["Payment_Behaviour"] = df_test["Payment_Behaviour"].replace("!@9#%8", np.nan)
df_test['Payment_Behaviour'] = df_test['Payment_Behaviour'].fillna(method='ffill')

df["Monthly_Balance"] = df["Monthly_Balance"].str.replace(r'[^0-9.-]+', '').str.replace('_', '').str.replace('-', '')
df['Monthly_Balance'] = df['Monthly_Balance'].astype(float)
mean_value = df['Monthly_Balance'].mean()
df['Monthly_Balance'].fillna(mean_value, inplace=True)
df_test["Monthly_Balance"] = df_test["Monthly_Balance"].str.replace(r'[^0-9.-]+', '')\
    .str.replace('_', '').str.replace('-', '')
df_test['Monthly_Balance'] = df_test['Monthly_Balance'].astype(float)
mean_value = df_test['Monthly_Balance'].mean()
df_test['Monthly_Balance'].fillna(mean_value, inplace=True)

df_num = df.select_dtypes(include='number')
for column in df_num.columns:
    for i in df["Credit_Score"].unique():
        selected_i = df[df["Credit_Score"] == i]
        selected_column = selected_i[column]

        std = selected_column.std()
        mean = selected_column.mean()

        max = mean + (4 * std)
        min = mean - (4 * std)

        outliers = selected_column[((selected_i[column] > max) | (selected_i[column] < min))].index
        df.drop(index=outliers, inplace=True)

columns_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month']
df = df.drop(columns=columns_drop)
df_test = df_test.drop(columns=columns_drop)

label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    df[column] = df[column].astype(str)
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

# Random Forest
rf_clf = RandomForestClassifier(random_state=66)
param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175, 200],'max_depth': [None, 10, 20, 30],}

grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best n_estimators:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

y_pred_rf = best_rf.predict(X_test)

report_rf = classification_report(y_test, y_pred_rf, target_names=label_encoders['Credit_Score'].classes_)
print(report_rf)

results = pd.DataFrame(grid_search.cv_results_)

plt.figure(figsize=(10, 6))
for depth in param_grid['max_depth']:
    subset = results[results['param_max_depth'] == depth]
    plt.plot(subset['param_n_estimators'], subset['mean_test_score'], label=f'max_depth={depth}')

plt.xlabel('n_estimators')
plt.ylabel('Mean CV Accuracy')
plt.title('Random Forest: n_estimators vs Accuracy for different max_depth')
plt.legend()
plt.grid(True)
plt.show()

importances = best_rf.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)
print()

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='royalblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()
plt.show()

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=66)
log_reg.fit(X_train, y_train)

y_pred_log_reg = log_reg.predict(X_test)

report_log_reg = classification_report(y_test, y_pred_log_reg, target_names=label_encoders['Credit_Score'].classes_)
print(report_log_reg)

# SCALER
scaler = StandardScaler()
X_train_sca = scaler.fit_transform(X_train)
X_test_sca = scaler.transform(X_test)

# SVM
svm_model = SVC(kernel='rbf', random_state=66)
svm_model.fit(X_train_sca, y_train)

y_pred_svm = svm_model.predict(X_test_sca)

report_svm = classification_report(y_test, y_pred_svm, target_names=label_encoders['Credit_Score'].classes_)
print(report_svm)

#GBDT
gbdt_model = GradientBoostingClassifier(random_state=66)
gbdt_model.fit(X_train_sca, y_train)

y_pred_gbdt = gbdt_model.predict(X_test_sca)

report_gbdt = classification_report(y_test, y_pred_gbdt, target_names=label_encoders['Credit_Score'].classes_)
print(report_gbdt)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

report_knn = classification_report(y_test, y_pred_knn, target_names=label_encoders['Credit_Score'].classes_)
print(report_knn)