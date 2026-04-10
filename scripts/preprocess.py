import pandas as pd

df = pd.read_csv("data/dataset.csv")

# remove missing values
df = df.dropna()

# remove duplicates
df = df.drop_duplicates()

# save cleaned data
df.to_csv("data/cleaned_dataset.csv", index=False)

print("Preprocessing completed")