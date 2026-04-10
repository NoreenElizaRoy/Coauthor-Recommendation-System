import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

import streamlit as st
from scripts.recommend import recommend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

# Page config
st.set_page_config(page_title="Collaboration Recommender", layout="wide")

# Title
st.title("🔍 Research Collaboration Recommendation System")

st.markdown("Find potential collaborators based on your research interest using AI & network analysis.")

# Sidebar
st.sidebar.header("Options")
show_clusters = st.sidebar.checkbox("Show Research Clusters")

# Input
topic = st.text_input("Enter Research Topic")

# Recommendation
if st.button("Recommend Collaborators"):

    results = recommend(topic)

    st.subheader("Top Recommended Researchers")

    for i, (author, score) in enumerate(results, 1):
        st.write(f"{i}. {author} — Similarity: {round(score,3)}")

# ---------------------------
# Cluster Visualization
# ---------------------------
if show_clusters:

    st.subheader("Researcher Clusters Visualization")

    with open("models/embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    with open("models/clusters.pkl", "rb") as f:
        cluster_data = pickle.load(f)

    embeddings = data["embeddings"]
    labels = [label for _, label in cluster_data]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels)

    ax.set_title("Clusters of Researchers")
    st.pyplot(fig)