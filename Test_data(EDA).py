#importing the required libraries for the EDA
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
#Importing the Dataset and performing the merge operation for EDA 
test = pd.read_csv('test.csv', nrows =2000)
label = pd.read_csv('test_label.csv', nrows =2000 )
merge = test.merge(label,how= 'outer' ,on ='0' )
df = merge.copy()


df.index
#Finding the informatin abou the dataset
df.info()
#Finding the description abiut the dataset
print(df.describe())

# Checking for missing values
print(df.isnull().sum())

# Plot histograms for numerical columns
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()
#Performing standardization on the dataset for easy visualization
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(df)

# Convert scaled_data back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

#ploting the pair plot for the Dataset 
import seaborn as sns

# Create a pairplot for numerical columns
sns.pairplot(scaled_df)
plt.show()
# Plotting the Co-realtion graph for indentifying the corealation between the features
correlation_matrix = df.corr()

# Visualize correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
# Listing  highly correlated features 
highly_correlated = correlation_matrix[abs(correlation_matrix) > 0.7].stack().reset_index()
highly_correlated = highly_correlated[highly_correlated['level_0'] != highly_correlated['level_1']]
print("Highly Correlated Features:")
print(highly_correlated)
#For deeper analysis of the dataset ploting the ploty graph for the dataset
import plotly.express as px

# Example: Scatter plot with hover tooltips
fig = px.scatter(df, x='1', y='2', color='0', hover_data=df.columns)
fig.update_layout(title='Interactive Scatter Plot', xaxis_title='Feature 1', yaxis_title='Feature 2')
fig.show()