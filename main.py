# Importing DataSet
import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
print("Path to dataset files:", path)

listDf = []
for file in os.listdir(path):
    listDf.append(pd.read_csv(os.path.join(path, file)))
dataset = pd.concat(listDf, ignore_index=True)
print(dataset.head())

dataset = dataset.fillna(dataset.mean())
print(dataset.head())