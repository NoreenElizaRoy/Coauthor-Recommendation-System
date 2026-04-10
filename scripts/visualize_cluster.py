import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load embeddings
with open("models/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

# Load cluster labels
with open("models/clusters.pkl", "rb") as f:
    cluster_data = pickle.load(f)

embeddings = data["embeddings"]
labels = [label for _, label in cluster_data]

# Reduce dimensions (important for plotting)
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=labels)

plt.title("Researcher Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.show()


#each color - one research domain
#close points - similar research interests