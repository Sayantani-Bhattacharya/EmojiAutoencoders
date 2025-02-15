import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("hf://datasets/valhalla/emoji-dataset/data/train-00000-of-00001-38cc4fa96c139e86.parquet")

# Visualise the dataframe - df

# Show the first few rows
print(df.head())

# List the column names
print(df.columns)

# Get a concise summary of the DataFrame
print(df.info())

# For a statistical summary of numeric columns
print(df.describe())