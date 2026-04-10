import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

df = pd.read_csv("data/cleaned_profiles.csv")

authors = df["author"].tolist()
profiles = df["title"].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(profiles)

data = {
    "authors": authors,
    "embeddings": embeddings
}

with open("models/embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Embeddings generated")