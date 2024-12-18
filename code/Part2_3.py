# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load the dataset
train_EDA = pd.read_csv('application_train.csv')
print(train_EDA.shape)

# Select specific columns
selected_columns = [
    'TARGET', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
    'AMT_INCOME_TOTAL', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'OCCUPATION_TYPE', 
    'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT_W_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_3'
]
train_EDA = train_EDA[selected_columns]
print(train_EDA.shape)

# Calculate the count and proportion of TARGET
counts = train_EDA['TARGET'].value_counts()
proportions = train_EDA['TARGET'].value_counts() / len(train_EDA)
result = pd.DataFrame({
    'Count': counts,
    'Proportion': proportions
})
print(result)

# Display the first few rows of the dataframe
print(train_EDA.head())

# Plot the distribution of TARGET
plt.figure(figsize=(6,6))
sns.countplot(data=train_EDA, x="TARGET", palette=["#e7bcbc", "#8d1d17"], width=0.4)
plt.xticks(ticks=[0, 1], labels=["0", "1"], fontsize=12)
plt.gca().set_xlim(-0.3, 1.3)
plt.ylabel('Count')
plt.xlabel('Target')
plt.tight_layout()
plt.savefig('plots/target_variable_distribution.png', dpi=300)
plt.show()

# Boxplot of OWN_CAR_AGE by FLAG_OWN_CAR
plt.figure(figsize=(6, 6))
sns.boxplot(data=train_EDA, x='FLAG_OWN_CAR', y='OWN_CAR_AGE', palette=["#c1272d", "#e7bcbc"])
plt.xlabel('Car Ownership')
plt.ylabel('Car Age (Years)')
plt.xticks([0, 1], ['No', 'Yes'])
plt.grid(axis='y')
plt.savefig('plots/Car Age by CarOwnership.png', dpi=300)
plt.show()

# Missing values summary for OWN_CAR_AGE and FLAG_OWN_CAR
missing_summary = train_EDA.groupby('FLAG_OWN_CAR')['OWN_CAR_AGE'].apply(lambda x: x.isnull().sum()).reset_index()
missing_summary.columns = ['Car Ownership (FLAG_OWN_CAR)', 'Count of Missing OWN_CAR_AGE']
total_counts = train_EDA['FLAG_OWN_CAR'].value_counts().reset_index()
total_counts.columns = ['Car Ownership (FLAG_OWN_CAR)', 'Total Count']
missing_summary = missing_summary.merge(total_counts, on='Car Ownership (FLAG_OWN_CAR)')
missing_summary['Percentage of Missing Values (%)'] = (missing_summary['Count of Missing OWN_CAR_AGE'] / missing_summary['Total Count']) * 100
print(missing_summary)

# Statistical description of DAYS_EMPLOYED
day_emp = train_EDA['DAYS_EMPLOYED'].describe()
day_emp_table = pd.DataFrame(day_emp)
day_emp_table.columns = ['DAYS_EMPLOYED']
print(day_emp_table)

# Anomaly analysis for DAYS_EMPLOYED
anom = train_EDA[train_EDA['DAYS_EMPLOYED'] == 365243]
non_anom = train_EDA[train_EDA['DAYS_EMPLOYED'] != 365243]
non_anom_default_rate = 100 * non_anom['TARGET'].mean()
anom_default_rate = 100 * anom['TARGET'].mean()
anom_count = len(anom)
summary_df = pd.DataFrame({
    'Category': ['Non-Anomalies', 'Anomalies'],
    'Default Rate (%)': [non_anom_default_rate, anom_default_rate],
    'Count of Anomalous Days of Employment': [0, anom_count] 
})
print(summary_df)

# Histogram of DAYS_EMPLOYED
plt.figure(figsize=(10, 6))
sns.histplot(train_EDA['DAYS_EMPLOYED'], bins=50, color="#e7bcbc", edgecolor='none')
plt.xlabel('Days Employed')
plt.ylabel('Count')
plt.savefig('plots/Distribution of Days Employed.png', dpi=300)
plt.show()

# Updated histogram of DAYS_EMPLOYED after replacing anomaly
train_EDA['DAYS_EMPLOYED_ANOM'] = train_EDA["DAYS_EMPLOYED"] == 365243
train_EDA['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
plt.figure(figsize=(10, 6))
sns.histplot(train_EDA['DAYS_EMPLOYED'], bins=50, color="#e7bcbc", edgecolor='none')
plt.xlabel('Days Employed')
plt.ylabel('Count')
plt.savefig('plots/Updated Distribution of Days Employed.png', dpi=300)
plt.show()

# Rename EXT_SOURCE_3 to EXT_SOURCE
train_EDA.rename(columns={'EXT_SOURCE_3': 'EXT_SOURCE'}, inplace=True)

# Missing values table
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_val_table_ren_columns[
        mis_val_table_ren_columns['Missing Values'] > 0].sort_values('% of Total Values', ascending=False).round(4)
    print("Dataframe has " + str(df.shape[1]) + " columns.\nThere are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    return mis_val_table_ren_columns

pd.set_option('display.max_rows', None)
missing_values = missing_values_table(train_EDA)
print(missing_values)

# Missing values in OCCUPATION_TYPE and DAYS_EMPLOYED
occupation_missing = train_EDA['OCCUPATION_TYPE'].isnull().sum()
days_employed_missing = train_EDA['DAYS_EMPLOYED'].isnull().sum()

print(f'Missing values in OCCUPATION_TYPE: {occupation_missing}')
print(f'Missing values in DAYS_EMPLOYED: {days_employed_missing}')

# Crosstab of DAYS_EMPLOYED and OCCUPATION_TYPE missing values
train_EDA['OCCUPATION_TYPE_MISSING'] = train_EDA['OCCUPATION_TYPE'].isnull()
train_EDA['DAYS_EMPLOYED_MISSING'] = train_EDA['DAYS_EMPLOYED'].isnull()

missing_crosstab = pd.crosstab(train_EDA['OCCUPATION_TYPE_MISSING'], train_EDA['DAYS_EMPLOYED_MISSING'], 
                                rownames=['OCCUPATION_TYPE Missing'], colnames=['DAYS_EMPLOYED Missing'])

print(missing_crosstab)

#%%
# Grouping by OCCUPATION_TYPE and calculating the mean of DAYS_EMPLOYED
occupation_days_employed = train_EDA.groupby('OCCUPATION_TYPE')['DAYS_EMPLOYED'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=occupation_days_employed, x='OCCUPATION_TYPE', y='DAYS_EMPLOYED', palette='viridis')
plt.title('Average Days Employed by Occupation Type')
plt.xlabel('Occupation Type')
plt.ylabel('Average Days Employed')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


#%%
# train_EDA = pd.read_csv('train_EDA.csv')


correlation_matrix = train_EDA.drop(columns=['TARGET','OCCUPATION_TYPE_MISSING','DAYS_EMPLOYED_MISSING']).corr()


plt.figure(figsize=(12, 10))
cmap = sns.color_palette("RdBu_r", as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, square=True,
            center=0, cbar_kws={"shrink": .8}, linewidths=0.5)


plt.title('Correlation Matrix of Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout() 
plt.savefig('plots/orrelation Matrix of Features.png', dpi=300)
plt.show()

print(correlation_matrix)