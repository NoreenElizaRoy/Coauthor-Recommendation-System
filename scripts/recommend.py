# import pickle
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import Counter

# # Load model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Load data
# with open("models/embeddings.pkl", "rb") as f:
#     data = pickle.load(f)

# with open("models/clusters.pkl", "rb") as f:
#     cluster_data = pickle.load(f)

# authors = data["authors"]
# embeddings = data["embeddings"]
# profiles = data["profiles"]   # ← NEW
# clusters = dict(cluster_data)

# # Stopwords (simple)
# stopwords = {"the", "and", "of", "in", "for", "to"}

# # Keyword extractor
# def get_keywords(text, top_n=3):
#     words = [w for w in text.split() if w not in stopwords]
#     common = Counter(words).most_common(top_n)
#     return [word for word, _ in common]


# def recommend(topic, top_k=10):

#     # Convert topic → embedding
#     topic_embedding = model.encode([topic])

#     # Compute similarity
#     scores = cosine_similarity(topic_embedding, embeddings)[0]

#     # Get best match cluster
#     best_index = scores.argmax()
#     best_author = authors[best_index]
#     target_cluster = clusters[best_author]

#     # Filter authors in same cluster
#     filtered = [
#         (authors[i], scores[i], i)
#         for i in range(len(authors))
#         if clusters[authors[i]] == target_cluster
#     ]

#     # Sort
#     ranked = sorted(filtered, key=lambda x: x[1], reverse=True)

#     # Add keywords
#     results = []
#     for author, score, idx in ranked[:top_k]:
#         keywords = get_keywords(profiles[idx])
#         results.append((author, score, keywords))

#     return results




# # import pickle
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity

# # model = SentenceTransformer('all-MiniLM-L6-v2')

# # with open("models/embeddings.pkl", "rb") as f:
# #     data = pickle.load(f)

# # with open("models/clusters.pkl", "rb") as f:
# #     cluster_data = pickle.load(f)

# # authors = data["authors"]
# # embeddings = data["embeddings"]
# # clusters = dict(cluster_data)


# # def recommend(topic, top_k=10):

# #     topic_embedding = model.encode([topic])

# #     scores = cosine_similarity(topic_embedding, embeddings)[0]

# #     # Get cluster of best match
# #     best_index = scores.argmax()
# #     best_author = authors[best_index]
# #     target_cluster = clusters[best_author]

# #     # Filter authors in same cluster
# #     filtered = [
# #         (authors[i], scores[i])
# #         for i in range(len(authors))
# #         if clusters[authors[i]] == target_cluster
# #     ]

# #     ranked = sorted(filtered, key=lambda x: x[1], reverse=True)

# #     return ranked[:top_k] how to modify this recommend

import pickle
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scripts.author_data import get_author_papers, get_collab_papers

# -----------------------------
# Load Model
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Load Data
# -----------------------------
with open("models/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

with open("models/clusters.pkl", "rb") as f:
    cluster_data = pickle.load(f)

authors = data["authors"]
embeddings = data["embeddings"]
profiles = data["profiles"]
titles = data.get("title", None)   # 👈 correct key   
clusters = dict(cluster_data)


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(topic, top_k=3):

    topic_embedding = model.encode([topic])
    scores = cosine_similarity(topic_embedding, embeddings)[0]

    best_index = scores.argmax()
    best_author = authors[best_index]

    target_cluster = clusters.get(best_author)

    # cluster filtering
    if target_cluster is None:
        filtered = [(authors[i], scores[i], i) for i in range(len(authors))]
    else:
        filtered = [
            (authors[i], scores[i], i)
            for i in range(len(authors))
            if clusters.get(authors[i]) == target_cluster
        ]

    # fallback
    if len(filtered) == 0:
        filtered = [(authors[i], scores[i], i) for i in range(len(authors))]

    ranked = sorted(filtered, key=lambda x: x[1], reverse=True)

    results = []
    for author, score, idx in ranked[:top_k]:

        own_papers = get_author_papers(author)
        collab_papers = get_collab_papers(author)

        results.append({
        "author": author,
        "score": float(score),
        "own_papers": own_papers,
        "collab_papers": collab_papers
    })
        

    return results