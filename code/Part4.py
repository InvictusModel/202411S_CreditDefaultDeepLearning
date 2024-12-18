# Import necessary libraries
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, EditedNearestNeighbours
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, mean_squared_error
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Load the dataset
df0 = pd.read_csv('GiveMeSomeCredit/cs-training.csv')
df0 = df0.dropna()

# Display the first few rows of the dataframe
print(df0.head())

# Model Integration
df_p = df0.copy()
nrow, ncol = df_p.shape
print(df_p.columns)

# Split the dataset into features and target variable
X = df_p.iloc[:, 1:-4]
y = df_p.iloc[:, 0]

# Split the dataset into training and testing sets
Xtr1, Xtt1, ytr1, ytt1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply random oversampling
ros = RandomOverSampler(random_state=42)
Xtr1, ytr1 = ros.fit_resample(Xtr1, ytr1)

# Add constant to the dataset for statsmodels
Xtr1 = sm.add_constant(Xtr1)
Xtt1 = sm.add_constant(Xtt1)

# Train the P model using OLS
def train_P_model(X, y):
    global P_model
    P_model = sm.OLS(y, X).fit()
    xcol = X.iloc[:, 1:].columns.tolist()
    print(P_model.summary(xname=["Intercept"] + xcol))
    return P_model

def predict_P(model, cases):
    yp = model.predict(cases)
    print(f"Predicted values --> {yp}")
    return yp

train_P_model(Xtr1, ytr1)
coefficients_P = P_model.params
print(f'Coefficients:{coefficients_P}')

# Predict using the test set
yp1 = predict_P(P_model, Xtt1)
print(yp1)

# EAD Model Integration
df_EAD = df0.copy()
nrow, ncol = df_EAD.shape
print(df_EAD.columns)

X = df_EAD.iloc[:, 1:11] 
y = df_EAD.iloc[:, -1]

# Split the dataset into training and testing sets
Xtr2, Xtt2, ytr2, ytt2 = train_test_split(X, y, test_size=0.2, random_state=42)

Xtr2 = sm.add_constant(Xtr2)
Xtt2 = sm.add_constant(Xtt2)

# Train the EAD model using OLS
def train_EAD_model(X, y):
    global EAD_model
    EAD_model = sm.OLS(y, X).fit()
    xcol = X.iloc[:, 1:].columns.tolist()
    print(EAD_model.summary(xname=["Intercept"] + xcol))
    return EAD_model

def predict_EAD(model, cases):
    yp = model.predict(cases)
    print(f"Predicted values --> {yp}")
    return yp

train_EAD_model(Xtr2, ytr2)
coefficients_EAD = EAD_model.params
print("Coefficients:")
print(coefficients_EAD)

# Predict using the test set
yp2 = predict_EAD(EAD_model, Xtt2)
print('LossRate:')
print(yp2)

# Prediction and Optimization
# Define the CF coefficients
n = 12
i = 0.04

coeff1 = i * (1 + i)**n / ((1 + i)**n - 1)
coeff2 = sum([1 / ((1 + i)**t) for t in range(1, n + 1)])
coeff3 = 1 / ((1 + i)**n)

print(coeff1, coeff2)

# Assume we have the data 'cases'
case = pd.read_csv('GiveMeSomeCredit/cs-training.csv')
case.reset_index(inplace=True)
case.rename(columns={'index': 'ID'}, inplace=True) # Each case has an ID
case.dropna(inplace=True)

# Drop unnecessary, create new
case.drop(['SeriousDlqin2yrs'], axis=1, inplace=True)
case['Loc'] = case.NumberOfOpenCreditLinesAndLoans * 700
case['LocUseRate'] = 1 - case.RevolvingUtilizationOfUnsecuredLines
case['LocUsage'] = case.Loc * case.LocUseRate
case['CF'] = case.LocUsage * coeff1 # Monthly cash flow
case.drop(['NumberOfOpenCreditLinesAndLoans'], axis=1, inplace=True)

case

# Collect variables
def wrap_row(row):
    wrap = [
        row['RevolvingUtilizationOfUnsecuredLines'],
        row['DebtRatio'],
        row['NumberOfTime30-59DaysPastDueNotWorse'],
        row['NumberOfTime60-89DaysPastDueNotWorse'],
        row['NumberOfTimes90DaysLate'],
        row['age'],
        row['MonthlyIncome'],
        row['NumberRealEstateLoansOrLines'],
        row['NumberOfDependents']
    ]
    return wrap

def collect_v(df):
    df['Variables'] = df.apply(lambda row: wrap_row(row), axis=1)
    return df

case = collect_v(case)
case

# Calculate k3_1 and k3_2
case['k3_1'] = case.apply(lambda row: np.dot(coefficients_P.values, np.append(1, row.Variables)), axis=1)
case['k3_2'] = case.apply(lambda row: np.dot(coefficients_EAD.values, np.append(1, row.Variables)), axis=1)
case['k3'] = k2 * case.k3_1 + k2 * case.k3_2
case['k4'] = case.LocUsage * coeff1 * coeff2
case['Q2'] = coeff3 * case.k3 - case.k4
case['Q3'] = case.k3_1 * case.k3_2

case

# Define the NPV function
def NPV(opt_Loc, Q1, Q2, Q3):
    x = opt_Loc
    npv = (-1) * Q1 * x**2 + Q2 * x - Q3
    return npv

# Define the optimization function
def optmz(row):
    Q2 = row['Q2']
    Q3 = row['Q3']

    f = lambda x: Q1 * x**2 - Q2 * x
    K1 = lambda x: (-1) * Q1 * x**2 + Q2 * x - Q3

    bounds = [(0, 5000000)]
    x0 = 0.01

    opt_Loc = scipy.optimize.minimize(f, x0, method='SLSQP', bounds=bounds, constraints=None).x
    max_NPV = NPV(opt_Loc, Q1, Q2, Q3)
    return opt_Loc, max_NPV

# Example usage
# case['opt_Loc'], case['max_NPV'] = zip(*case.apply(lambda row: optmz(row), axis=1))

# Save the optimized results to a CSV file
case.to_csv('Credit_NPV_solved.csv', index=True)