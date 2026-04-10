import pandas as pd
import networkx as nx
import pickle

df = pd.read_csv("data/cleaned_dataset.csv")

G = nx.Graph()

for title, group in df.groupby("title"):

    authors = list(group["author"])

    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            G.add_edge(authors[i], authors[j])

with open("models/network.pkl", "wb") as f:
    pickle.dump(G, f)

print("Co-authorship network created")