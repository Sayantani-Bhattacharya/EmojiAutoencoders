import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("hf://datasets/valhalla/emoji-dataset/data/train-00000-of-00001-38cc4fa96c139e86.parquet")

# Visualising the dataframe.

# # Show the first few rows
# print(df.head())
# # List the column names
# print(df.columns)
# # Get a concise summary of the DataFrame
# print(df.info())
# # For a statistical summary of numeric columns
# print(df.describe())
# Get all the sub-group options
print(df["text"].unique())

# Pre-processing the data.

# Step1: Text based filtering.
face_emojis = df[df["text"].str.contains("face", case=False, na=False)]
print(face_emojis.info())
baby_angel = df[df["text"].str.contains("baby angel", case=False, na=False)]
superhero = df[df["text"].str.contains("superhero", case=False, na=False)]


# Step2: Divide this subset into training, validation and test sets using a 60/20/20 ratio.
train_size = int(len(face_emojis) * 0.6)
val_size = int(len(face_emojis) * 0.2)
train_set = face_emojis[:train_size]
val_set = face_emojis[train_size:train_size + val_size]
test_set = face_emojis[train_size + val_size:]

# Step3: Data augmentation by adding generated adversarial examples.
# Using pyTorch's Torchattacks to generate adversarial examples.
