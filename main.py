import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
import kagglehub
import mlflow
import tensorflow as tf
import mlflow.sklearn


# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Import the data from Kaggle
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
csv_file = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
data = pd.read_csv(csv_file)

import tensorflow as tf 


# Basic data insights
print("Dataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())


mean_bmi = data['BMI'].mean()
median_bmi = data['BMI'].median()

# Plot the histogram with mean,, and median
plt.figure(figsize=(8, 6))
sns.histplot(data['BMI'], kde=True, bins=30, color='blue')

# Add vertical lines for mean, mode, and median
plt.axvline(mean_bmi, color='r', linestyle='--', label='Mean')
plt.axvline(median_bmi, color='b', linestyle='--', label='Median')

# Customize the plot
plt.title('Distribution of BMI', fontsize=14)
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.show()
sns.boxplot(
    x = "Diabetes_012",
    y = "BMI",
    showmeans=True,
    data=data
)

# Customize the plot
plt.title("Distribution of BMI by Diabetes Diagnosis")
plt.xlabel("Diabetes Diagnosis ")
plt.ylabel("Body Mass Index (BMI)")
plt.show()

plt.title('Distribution of BMI', fontsize=14)
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
sns.boxplot(
    x = "Diabetes_012",
    y = "BMI",
    showmeans=True,
    data=data
)

Q1 = data['BMI'].quantile(0.25)
Q3 = data['BMI'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = data[(data['BMI'] < lower_bound) | (data['BMI'] > upper_bound)]

# Define bins for histogram
bins = range(int(data['BMI'].min()), int(data['BMI'].max()) + 1, 1)

# Plot histogram for BMI including outliers
plt.figure(figsize=(8, 6))
plt.hist(data['BMI'], bins=bins, alpha=0.5, label='Normal')
plt.hist(outliers['BMI'], bins=bins, alpha=0.5, label='Outliers', color='r')
plt.legend()
plt.title('BMI Histogram: Outliers vs Normal')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()

# Heatmap for correlation
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Extracting highly correlated pairs
threshold = 0.4
correlated_pairs = [
    (col, row, correlation_matrix.loc[row, col])
    for col in correlation_matrix.columns
    for row in correlation_matrix.index
    if col != row and abs(correlation_matrix.loc[row, col]) >= threshold
]

print(f"\nCorrelated variables with correlation >= {threshold}:")
unique_pairs = {}
for pair in correlated_pairs:
    pair_key = tuple(sorted([pair[0], pair[1]]))
    if pair_key not in unique_pairs:
        unique_pairs[pair_key] = pair[2]

for (var1, var2), corr in unique_pairs.items():
    print(f"{var1} and {var2} have a correlation of {corr:.2f}")

# Prepare target and features
y = data['Diabetes_012']
X = data.drop('Diabetes_012', axis=1)

# Balance the dataset using oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Display class distribution after oversampling
print("\nClass distribution after oversampling:")
print(pd.Series(y_resampled).value_counts())

# Normalize specific columns
columns_to_normalize = ['BMI', 'MentHlth', 'GenHlth', 'PhysHlth', 'Age', 'Education', 'Income']
X_resampled[columns_to_normalize] = Normalizer().fit_transform(X_resampled[columns_to_normalize])



# Split the data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=122, stratify=y_resampled
)

# Step 2: Initialize Stratified K-Fold
n_splits = 2
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=122)

# Step 3: Iterate over each fold
fold_accuracies = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train_full, y_train_full), 1):
    print(f"\nFold {fold}:")

    # Use .iloc for proper indexing
    X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

    print(f"  Training size: {len(X_train)}, Validation size: {len(X_val)}")

    # Step 4: Train a model (Random Forest in this case)
    model = RandomForestClassifier(random_state=122)
    model.fit(X_train, y_train)

    # Step 5: Validate the model
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(accuracy)
    
    print(f"  Fold {fold} Validation Accuracy: {accuracy:.4f}")

# Step 6: Evaluate on the Test Set (optional)
final_model = RandomForestClassifier(random_state=122)
final_model.fit(X_train_full, y_train_full)
y_test_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nCross-Validation Accuracies: {fold_accuracies}")
print(f"Mean Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Test Set Accuracy: {test_accuracy:.4f}")

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Log model and metrics
        mlflow.log_param("model_type", model_name)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report
        report = classification_report(y_test, predictions, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log model
        mlflow.sklearn.log_model(model, f"{model_name}_model")

        # Print results
        print(f"\n{model_name} Performance:")
        print("Accuracy:", accuracy)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Visualize confusion matrix
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
        plt.title(f"Confusion Matrix: {model_name}")
        plt.show()

dt_model = DecisionTreeClassifier(random_state=42)
train_and_log_model(dt_model, "Decision_Tree_Classifier", X_train, X_test, y_train, y_test)

rf_model = RandomForestClassifier(random_state=42)
train_and_log_model(rf_model, "Random_Forest_Classifier", X_train, X_test, y_train, y_test)

# export the  best model rf

import pickle

# Save the Random Forest model to a file
with open("random_forest_diabetes_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)
print("Random Forest model saved successfully!")


with open("random_forest_diabetes_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)
print("Model loaded successfully!")

import numpy as np

# Example new data (replace these values with real input data)
new_individual = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 36,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 0,
    "Fruits":1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 3,
    "MentHlth": 7,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Sex": 0,
    "Age": 9,
    "Education": 5,
    "Income": 4
}


# Convert input to DataFrame with the same structure as training data
new_data_df = pd.DataFrame([new_individual])

# Apply normalization (ensure the same preprocessing steps as training)
columns_to_normalize = ['BMI', 'MentHlth', 'GenHlth', 'PhysHlth', 'Age', 'Education', 'Income']
normalizer = Normalizer()
new_data_df[columns_to_normalize] = normalizer.fit_transform(new_data_df[columns_to_normalize])

# Predict using the loaded model
prediction = loaded_model.predict(new_data_df)
print("Prediction (0 = No Diabetes, 1 = Diabetes, 2 = Prediabetes):", prediction[0])
