#Importing the Libraries

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

#Importing the dataset and merging the datasets
test = pd.read_csv('psm_test.csv', nrows =2000)
label = pd.read_csv('psm_test_label.csv', nrows =2000 )
merge = test.merge(label,how= 'outer' ,on ='timestamp_(min)' )
df = merge.copy()
df.index
df.info()

#converting The timestamp_(min) into datetime format
df['timestamp_(min)'] = pd.to_datetime(df['timestamp_(min)'])
#Setting the timestamp as a index of the dataset
df.set_index('timestamp_(min)', inplace=True)
#highlighting the Anomaly in the dataset 
features_of_interest = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24']
window_size = 10  
rolling_means = df[features_of_interest].rolling(window=window_size).mean()
rolling_stds = df[features_of_interest].rolling(window=window_size).std()
anomaly_thresholds = rolling_means + (2 * rolling_stds)
anomalies = {feature: df[df[feature] > anomaly_thresholds[feature]] for feature in features_of_interest}

#Plotting the Anomaly in the dataset
plt.figure(figsize=(12, 8))

for feature in features_of_interest:
    plt.plot(df.index, df[feature], label=feature)
    plt.fill_between(anomaly_thresholds.index, anomaly_thresholds[feature], color='red', alpha=0.3, label=f'{feature} Anomaly Region')

plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Time Series Data with Anomaly Regions for Multiple Features')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#scaling the dataset for easy visualization
scaler = StandardScaler()
# Fit and transform the data
scaled_data = scaler.fit_transform(df)
# Convert scaled_data back to DataFrame (optional)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
#Anomaly percentage in the dataset
model = IsolationForest(contamination=0.05)  
model.fit(df)
anomaly_scores = model.decision_function(df)
anomaly_mask = model.predict(df)

plt.figure(figsize=(12, 8))
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

# Highlight Anomalies
plt.scatter(df.index, df[df.columns[0]], c=anomaly_mask, cmap='viridis', label='Anomaly')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Time Series Data with Anomalies')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#feature decompostion graph of the dataset
features_to_decompose = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24']
  # Add more features as needed

# Perform seasonal decomposition for each feature
plt.figure(figsize=(12, 8))
for feature in features_to_decompose:
    result = seasonal_decompose(df[feature], model='additive', period=12)  # Adjust period based on data seasonality

    # Plot decomposed components
    plt.subplot(len(features_to_decompose), 1, features_to_decompose.index(feature) + 1)
    plt.plot(df[feature], label='Original')
    plt.plot(result.trend, label='Trend')
    plt.plot(result.seasonal, label='Seasonal')
    plt.plot(result.resid, label='Residual')
    plt.title(f'Time Series Decomposition: {feature}')
    plt.legend()

plt.tight_layout()
plt.show()

#plotting the colrelation graph between feature and finding highly correlated feature 
import seaborn as sns
correlation_matrix = df.corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# List highly correlated features (absolute correlation > 0.7 for example)
highly_correlated = correlation_matrix[abs(correlation_matrix) > 0.7].stack().reset_index()
highly_correlated = highly_correlated[highly_correlated['level_0'] != highly_correlated['level_1']]
print("Highly Correlated Features:")
print(highly_correlated)

