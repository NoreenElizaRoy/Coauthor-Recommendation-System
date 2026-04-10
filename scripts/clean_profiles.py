import pandas as pd
import re

df = pd.read_csv("data/cleaned_dataset.csv")

# Combine titles per author
profiles = df.groupby("author")["title"].apply(lambda x: " ".join(x))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)  # remove symbols
    text = re.sub(r'\s+', ' ', text)        # remove extra spaces
    return text

cleaned_profiles = profiles.apply(clean_text)

# Save cleaned profiles
cleaned_profiles.to_csv("data/cleaned_profiles.csv")

print("Profiles cleaned successfully")