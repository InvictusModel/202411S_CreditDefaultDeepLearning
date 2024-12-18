# Import the necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/Users/xxx/Desktop/accepted_2007_to_2018Q4.csv')
small_data = data.sample(frac=0.1, random_state=123)
small_data.to_csv('/Users/xxx/Desktop/LC.csv', index=False)

# Load the dataset (assuming LC is a CSV file)
df = pd.read_csv('/Users/xxx/Desktop/LC.csv')  # Adjust the file path if necessary

# Select the features you want to analyze for correlation
features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']

# Filter the dataset to include only the selected features
df_selected = df[features]

# Compute the correlation matrix
correlation_matrix = df_selected.corr()

# Display the correlation matrix
print(correlation_matrix)

# Plot a heatmap to visualize the correlation, using only red shades
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Selected Features')

# Save the figure as a PNG file
plt.savefig('/Users/xxx/Desktop/correlation_matrix_heatmap.png', dpi=300, bbox_inches='tight')

# Show the heatmap
plt.show()

#%%

# Load dataset
df = pd.read_csv('/Users/liujinming/Desktop/AN6003 Analytics Strategy/Dataset/LC.csv')

# Get basic statistics of the installment column
installment_stats = df['installment'].describe()
print(installment_stats)

# Plot histogram to visualize the distribution of installment amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['installment'], bins=30, kde=True)
plt.title('Distribution of Installment Amounts')
plt.xlabel('Installment Amount')
plt.ylabel('Frequency')
plt.show()

#%%
# Filter the DataFrame to include only 'Fully Paid' and 'Charged Off' loan statuses
df_filtered = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

# Calculate the count for each combination of installment group and loan status
group_counts = df_filtered.groupby(['installment_group', 'loan_status']).size().unstack(fill_value=0)

# Calculate the total number of loans in each installment group
group_totals = group_counts.sum(axis=1)

# Calculate the proportion for each loan status in each installment group
group_proportions = group_counts.div(group_totals, axis=0) * 100

# Display the proportions
print(group_proportions)