import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


#Importing the dataset and merging the datasets
test = pd.read_csv('msl_test.csv', nrows =2000)
label = pd.read_csv('msl_test_label.csv', nrows =2000 )
merge = test.merge(label,how= 'outer' ,on ='0' )
df = merge.copy()
df.index
df.info()

# Identify binary feature columns
binary_columns = [col for col in df.columns if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}]

# Print value counts for each binary column
for column in binary_columns:
    print(f"Value counts for {column}:")
    print(df[column].value_counts())
    print()  # Empty line for better readability


import seaborn as sns

# Plot count plots for binary features
for column in binary_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=column)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(f'Count Plot of {column}')
    plt.show()



#Correlation analysis between binary features
binary_df = df[binary_columns]
binary_corr = binary_df.corr()
print("Correlation Matrix between Binary Features:")
print(binary_corr)

from scipy.stats import zscore

#  Z-score anomaly detection for binary features
for column in binary_columns:
    z_scores = zscore(df[column])
    outliers = df[(z_scores > 3) | (z_scores < -3)]  # Adjust threshold as needed
    print(f"Number of outliers detected in {column}: {len(outliers)}")
    print(outliers)


# Example: Correlation analysis between binary features
binary_corr = df[binary_columns].corr()
print("Correlation Matrix between Binary Features:")
print(binary_corr)

