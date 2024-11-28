# Importing DataSet
import kagglehub
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
print("Path to dataset files:", path)

# Combine every file into one dataframe
listDf = []
for file in os.listdir(path):
    listDf.append(pd.read_csv(os.path.join(path, file)))
dataset = pd.concat(listDf, ignore_index=True)
print("\nRaw data:\n", dataset.head())

# Check for missing values -> fill with mean
dataset = dataset.fillna(dataset.mean())
print("\nNo missing values anymore (mean replacement):\n", dataset.head())

# Normalization
dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
print("\nNormalized data:\n", dataset.head())

# Compute correlation matrix
correlation_matrix = dataset.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()

# Define a threshold for "high" correlation
threshold = 0.3

# Iterate through the correlation matrix and avoid duplicates
correlated_pairs = []

for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        # Skip self-correlation and avoid duplicate pairs
        if col != row and col < row:
            correlation_value = correlation_matrix.loc[row, col]
            if abs(correlation_value) >= threshold:
                correlated_pairs.append((row, col, correlation_value))

# Print the correlated variable pairs
print(f"Correlated variables with correlation >= {threshold}:")
for pair in correlated_pairs:
    print(f"{pair[0]} and {pair[1]} have a correlation of {pair[2]:.2f}")
