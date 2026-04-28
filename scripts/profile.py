import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain


def show_profile(author_data, G):

    if not author_data:
        st.error("No author data found")
        return

    st.markdown("---")

    col1, col2 = st.columns([6,1])
    with col2:
        if st.button("❌ Close Profile"):
            st.session_state["show_profile"] = False

    st.header(f"👤 {author_data['author'].title()}")
    st.write(f"Similarity Score: {author_data['score']:.2f}")

    # 📄 Papers
    all_papers = author_data['own_papers'] + author_data['collab_papers']
    unique_papers = list(dict.fromkeys(all_papers))

    st.subheader("📄 Publications")

    for i, p in enumerate(unique_papers, 1):
        st.write(f"{i}. {p}")

    # 📊 Cluster
    st.subheader("📊 Collaboration Cluster")

    partition = community_louvain.best_partition(G)

    author_name = author_data['author'].lower()

    cluster_id = partition.get(author_name)

    if cluster_id is not None:

        cluster_members = [n for n, c in partition.items() if c == cluster_id]

        st.write(f"Cluster ID: {cluster_id}")
        st.write(f"Members: {len(cluster_members)}")

        for name in cluster_members:
            st.write("•", name.title())

        subG = G.subgraph(cluster_members)

        plt.figure(figsize=(6,4))
        pos = nx.spring_layout(subG, seed=42)

        nx.draw(subG, pos, with_labels=True, node_size=700, font_size=8)

        st.pyplot(plt)

    else:
        st.warning("No cluster found")