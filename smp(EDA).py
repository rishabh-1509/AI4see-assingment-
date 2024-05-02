#Importing the library
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

#Loading the dataset
test = pd.read_csv('smap_test.csv', nrows =10000)
label = pd.read_csv('smap_test_label.csv', nrows =10000 )
merge = test.merge(label,how= 'outer' ,on ='0' )
df = merge.copy()
df.index
df.info()

#descprition if the dataset
print(df.describe())

# Check for missing values
null_columns = df.columns[df.isnull().all()]
print("Null Columns:", null_columns)

# Drop null columns from the DataFrame
df.drop(null_columns, axis=1, inplace=True)

# saveing  the modified DataFrame to a new CSV file
df.to_csv('modified_dataset.csv', index=False)



# Plot histograms for numerical columns
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

#Performing the standard scaling for the Dataset
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(df)

# Convert scaled_data back to DataFrame (optional)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)


# Identify binary feature columns (assuming binary values are 0 and 1)
binary_columns = []
for column in df.columns:
    if df[column].nunique() == 2 and set(df[column].unique()) == {0, 1}:
        binary_columns.append(column)

for column in binary_columns:
    print(f"Value counts for {column}:")
    print(df[column].value_counts())
    print()


import seaborn as sns

# Count plot for binary features
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='0')
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Count Plot of Binary Feature 1')
plt.tight_layout()
plt.show()


from scipy.stats import zscore
binary_columns = []
for column in df.columns:
    if df[column].nunique() == 2 and set(df[column].unique()) == {0, 1}:
        binary_columns.append(column)

for column in binary_columns:
    print(f"Analysis for {column}:")
    
    # Calculate value counts
    value_counts = df[column].value_counts()
    print("Value Counts:")
    print(value_counts)
    
    # Visualize value counts
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=column)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'Count Plot of {column}')
    plt.show()
    

#Calculating the z score for columns for outlier identification
z_scores = zscore(df[column])
outliers = df[(z_scores > 2) | (z_scores < -2)]  
print(f"Number of outliers detected in {column}: {len(outliers)}")
print(outliers)  # Print the outliers for further inspection
print()  
print()  # Empty line for better readability


# Perform chi-square test for independence
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(cross_tab)
print(f"\nChi-Square Statistic: {chi2}")
print(f"P-value: {p}")
