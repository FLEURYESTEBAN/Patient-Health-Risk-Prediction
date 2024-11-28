# Importing Libraries
import kagglehub
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.decomposition import PCA

# Download latest version of the dataset
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
print("Path to dataset files:", path)

# Combine every file into one dataframe
listDf = [pd.read_csv(os.path.join(path, file)) for file in os.listdir(path)]
dataset = pd.concat(listDf, ignore_index=True)
print("\nRaw data:\n", dataset.head())

# Check for missing values -> fill with mean
dataset = dataset.fillna(dataset.mean())
print("\nNo missing values anymore (mean replacement):\n", dataset.head())

# Define features and apply scaling & normalization
features = dataset.drop(columns=['Diabetes_binary'], errors='ignore')  # Drop target column if exists
scaler = MinMaxScaler()
normalizer = Normalizer()

# Apply transformations
transformed_data = {
    "Original Data": features,
    "Min-Max Scaling": pd.DataFrame(scaler.fit_transform(features), columns=features.columns),
    "Normalization": pd.DataFrame(normalizer.fit_transform(features), columns=features.columns)
}

# PCA for visualization
pca_results = {}
for name, data in transformed_data.items():
    pca = PCA(n_components=2)
    pca_results[name] = pca.fit_transform(data)

# Visualize PCA transformations
fig, axes = plt.subplots(1, len(pca_results), figsize=(18, 6))
axes = axes.ravel()

for ax, (name, pca_data) in zip(axes, pca_results.items()):
    ax.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
    ax.set_title(name)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

plt.tight_layout()
plt.show()

# Compute correlation matrix
correlation_matrix = dataset.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Find highly correlated variables
threshold = 0.3
correlated_pairs = []

for i in correlation_matrix.columns:
    for j in correlation_matrix.index:
        if i != j and (j, i) not in correlated_pairs:  # Avoid duplicates
            if abs(correlation_matrix.loc[j, i]) >= threshold:
                correlated_pairs.append((i, j, correlation_matrix.loc[j, i]))

# Display correlated variable pairs
print(f"Correlated variables with correlation >= {threshold}:")
for var1, var2, corr in correlated_pairs:
    print(f"{var1} and {var2}: correlation {corr:.2f}")
