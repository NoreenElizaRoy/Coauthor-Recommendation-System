import pickle
from sklearn.cluster import KMeans

# Load embeddings
with open("models/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

authors = data["authors"]
embeddings = data["embeddings"]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0)
labels = kmeans.fit_predict(embeddings)

# Save clusters
cluster_data = list(zip(authors, labels))

with open("models/clusters.pkl", "wb") as f:
    pickle.dump(cluster_data, f)

print("Clustering completed")

