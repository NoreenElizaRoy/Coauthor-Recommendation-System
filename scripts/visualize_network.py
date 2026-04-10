import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Load graph
with open("models/network.pkl", "rb") as f:
    G = pickle.load(f)

plt.figure(figsize=(10, 8))

# Draw graph
nx.draw(G,
        with_labels=True,
        node_size=50,
        font_size=6)

plt.title("Co-Authorship Network")
plt.show()