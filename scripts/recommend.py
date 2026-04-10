import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("models/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

with open("models/clusters.pkl", "rb") as f:
    cluster_data = pickle.load(f)

authors = data["authors"]
embeddings = data["embeddings"]
clusters = dict(cluster_data)

def recommend(topic, top_k=10):

    topic_embedding = model.encode([topic])

    scores = cosine_similarity(topic_embedding, embeddings)[0]

    # Get cluster of best match
    best_index = scores.argmax()
    best_author = authors[best_index]
    target_cluster = clusters[best_author]

    # Filter authors in same cluster
    filtered = [
        (authors[i], scores[i])
        for i in range(len(authors))
        if clusters[authors[i]] == target_cluster
    ]

    ranked = sorted(filtered, key=lambda x: x[1], reverse=True)

    return ranked[:top_k]