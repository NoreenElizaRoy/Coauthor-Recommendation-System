# import networkx as nx
# import streamlit as st
# from scripts.recommend import recommend
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import pickle
# import streamlit.components.v1 as components
# from scripts.author_data import get_full_author_data
# import re
# # Page config
# st.set_page_config(page_title="Collaboration Recommender", layout="wide")

# # Title
# st.title("🔍 Research Collaboration Recommendation System")

# st.markdown("Find potential collaborators based on your research interest using AI & network analysis.")

# # Sidebar
# st.sidebar.header("Options")
# show_clusters = st.sidebar.checkbox("Show Research Clusters")
# show_network = st.sidebar.checkbox("Show Network Graph")

# # Input
# topic = st.text_input("Enter Research Topic")

# import streamlit as st
# import streamlit.components.v1 as components

# # results = []
# if "results" not in st.session_state:
#     st.session_state["results"] = []

# if st.button("Recommend Collaborators"):

#     # results = recommend(topic, top_k=3)
#     st.session_state["results"] = recommend(topic, top_k=3)
#     recommended_authors = [r["author"].lower().strip() for r in results]
#     results = st.session_state["results"]
#     st.subheader("Top Recommended Researchers")

#     if not results:
#         st.warning("No recommendations found")

#     else:
#         cols = st.columns(3)

#         for i, r in enumerate(results[:3]):
#             with cols[i]:
#                 # if st.button("View Profile", key=f"profile_{i}"):

#                 #     full_data = get_full_author_data(
#                 #         r["author"],
#                 #         r["score"]
#                 #         )

#                 #     st.session_state["selected_author"] = full_data
#                 #     st.session_state["show_profile"] = True


#                 all_papers = r['own_papers'] + r['collab_papers']
#                 unique_papers = list(dict.fromkeys(all_papers))[:3]

#                 border_color = "#FFD700" if i == 0 else "#4CAF50"
#                 bg_color = "#fffbea" if i == 0 else "#f9f9ff"

#                 # 🔥 Build papers with class
#                 papers_html = "<ul style='padding-left:18px; margin-top:10px;'>"
#                 for p in unique_papers:
#                     papers_html += f"""
#                     <li class="paper">
#                         {p}
#                     </li>
#                     """
#                 papers_html += "</ul>"

#                 # 🔥 Add hover CSS + card class
#                 html_code = f"""
#                 <style>
#                 .card {{
#                     transition: all 0.3s ease;
#                     cursor: pointer;
#                 }}

#                 .card:hover {{
#                     transform: translateY(-8px) scale(1.03);
#                     box-shadow: 0 10px 25px rgba(0,0,0,0.3);
#                 }}

#                 .paper {{
#                     margin-bottom:6px;
#                     padding:6px 8px;
#                     background:white;
#                     border-radius:6px;
#                     color:black;
#                     transition: all 0.2s ease;
#                 }}

#                 .paper:hover {{
#                     background:#e3f2fd;
#                     transform: translateX(5px);
#                 }}
#                 </style>

#                 <div class="card" style="
#                     border-left:6px solid {border_color};
#                     background:{bg_color};
#                     padding:15px;
#                     border-radius:15px;
#                     font-family:sans-serif;
#                     color:black;
#                     height:280px;
#                     overflow-y:auto;
#                 ">
#                     <h3 style="margin-bottom:5px;">
#                         {'🥇' if i==0 else '🥈' if i==1 else '🥉'} {r['author'].title()}
#                     </h3>

#                     <p style="margin-bottom:10px;">
#                         Similarity: {r['score']:.2f}
#                     </p>

#                     {papers_html}
#                 </div>
#                 """

#                 components.html(html_code, height=300)


# def normalize(name):
#     name = str(name).lower().strip()
#     name = name.replace("_", " ")
#     name = re.sub(r'[^a-z\s]', '', name)
#     return name


# if show_network:


#     st.subheader("🟢 Co-Authorship Network with Recommendations")

# # Load graph
#     with open("models/network.pkl", "rb") as f:
#         G = pickle.load(f)

# # Normalize graph node names
#     G = nx.relabel_nodes(G, normalize)

    

# # Get recommended authors
#     recommended_authors = [
#     normalize(r["author"]) for r in results
# ]
#     matches = [node for node in G.nodes() if node in recommended_authors]
#     st.write("Matched nodes:", matches)

#     st.write("Recommended Authors:", recommended_authors[:5])

# # Reduce graph size (important for clarity)
#     if len(G.nodes()) > 100:
#         G = G.subgraph(list(G.nodes())[:100])

# # Layout
#     pos = nx.spring_layout(G, seed=42)

# # Color nodes
#     node_colors = [
#         "red" if node in recommended_authors else "lightblue"
#         for node in G.nodes()
#     ]

# # Labels: show only for recommended authors
#     labels = {
#     node: node if node in recommended_authors else ""
#     for node in G.nodes()
# }    

# # Draw
#     fig, ax = plt.subplots(figsize=(8, 6))

#     nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     labels=labels,
#     node_color=node_colors,
#     node_size=300,
#     font_size=7,
#     edge_color="gray",
#     alpha=0.7,
#     ax=ax
#     )

#     ax.set_title("Co-Authorship Network ")

#     st.pyplot(fig)
#     st.write("Recommended:", recommended_authors[:3])
#     st.write("Graph nodes:", list(G.nodes())[:10])



import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import re
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from scripts.recommend import recommend
import community.community_louvain as community_louvain
from scripts.author_data import get_full_author_data

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Collaboration Recommender", layout="wide")

st.title("🔍 Research Collaboration Recommendation System")
st.markdown("Find potential collaborators based on your research interest using AI & network analysis.")

# ------------------ SIDEBAR ------------------
st.sidebar.header("Options")
show_network = st.sidebar.checkbox("Network Graph")
show_community = st.sidebar.checkbox("Communities")
# ------------------ INPUT ------------------
topic = st.text_input("Enter Research Topic")

# ------------------ SESSION STATE ------------------
if "results" not in st.session_state:
    st.session_state["results"] = []

# ------------------ NORMALIZATION ------------------
def normalize(name):
    name = str(name).lower().strip()
    name = name.replace("_", " ")
    name = re.sub(r'[^a-z\s]', '', name)
    return name

# ------------------ BUTTON ------------------
if st.button("Recommend Collaborators"):
    st.session_state["results"] = recommend(topic, top_k=3)

# ALWAYS FETCH RESULTS
results = st.session_state["results"]

# ------------------ DISPLAY RESULTS ------------------
st.subheader("Top Recommended Researchers")

if not results:
    st.info("Enter a topic and click 'Recommend Collaborators'")
else:
    cols = st.columns(3)

    for i, r in enumerate(results[:3]):
        with cols[i]:

            all_papers = r['own_papers'] + r['collab_papers']
            unique_papers = list(dict.fromkeys(all_papers))[:3]

            border_color = "#FFD700" if i == 0 else "#4CAF50"
            bg_color = "#fffbea" if i == 0 else "#f9f9ff"

            papers_html = "<ul style='padding-left:18px; margin-top:10px;'>"
            for p in unique_papers:
                papers_html += f"<li class='paper'>{p}</li>"
            papers_html += "</ul>"

            html_code = f"""
            <style>
            .card {{
                transition: all 0.3s ease;
                cursor: pointer;
            }}
            .card:hover {{
                transform: translateY(-8px) scale(1.03);
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            }}
            .paper {{
                margin-bottom:6px;
                padding:6px 8px;
                background:white;
                border-radius:6px;
                color:black;
            }}
            </style>

            <div class="card" style="
                border-left:6px solid {border_color};
                background:{bg_color};
                padding:15px;
                border-radius:15px;
                color:black;
                height:280px;
                overflow-y:auto;
            ">
                <h3>{'🥇' if i==0 else '🥈' if i==1 else '🥉'} {r['author'].title()}</h3>
                <p>Similarity: {r['score']:.2f}</p>
                {papers_html}
            </div>
            """

            components.html(html_code, height=300)

# ------------------ NETWORK GRAPH ------------------
if show_network:

    st.subheader("🟢 Interactive Co-Authorship Network")

    import plotly.graph_objects as go

    # ---------- Load graph ----------
    with open("models/network.pkl", "rb") as f:
        G = pickle.load(f)

    # ---------- Normalize ----------
    G = nx.relabel_nodes(G, normalize)

    # ---------- Recommendations ----------
    results = st.session_state.get("results", [])

    recommended_authors = [normalize(r["author"]) for r in results]
    matches = set(recommended_authors)

    # ---------- Limit graph ----------
    if len(G.nodes()) > 100:
        G = G.subgraph(list(G.nodes())[:100])

    pos = nx.spring_layout(G, seed=42)

    # ---------- EDGE ----------
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="gray"),
        hoverinfo="none"
    )

    # ---------- NODE ----------
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Label ALWAYS shown (no special hover formatting)
        node_text.append(node)

        if node in matches:
            node_color.append("red")   # highlight recommended
            node_size.append(18)
        else:
            node_color.append("lightblue")
            node_size.append(10)

    node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",   # 👈 shows labels directly
    text=node_text,
    textposition="top center",
    hoverinfo="none",      # 👈 disables hover popup completely
    marker=dict(
        size=node_size,
        color=node_color,
        line=dict(width=1)
    )
)

    # ---------- FIGURE ----------
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Co-Authorship Network (Interactive)",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

if show_community:

    st.subheader("🧠 Research Community Detection Network")

    # Load graph
    with open("models/network.pkl", "rb") as f:
        G = pickle.load(f)

    # Normalize nodes
    G = nx.relabel_nodes(G, normalize)

    # Limit size for performance
    if len(G.nodes()) > 120:
        G = G.subgraph(list(G.nodes())[:120])

    # ---------------- COMMUNITY DETECTION ----------------
    partition = community_louvain.best_partition(G)

    # Group nodes by community
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    # Create community names
    community_names = {}
    for comm_id, nodes in communities.items():
        # simple auto naming based on first node (can improve later)
        community_names[comm_id] = f"Research Group {comm_id+1}"

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # ---------------- EDGES ----------------
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="gray"),
        hoverinfo="none"
    )

    # ---------------- NODES ----------------
    node_x = []
    node_y = []
    node_color = []
    node_text = []

    colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
        "#f58231", "#911eb4", "#46f0f0", "#f032e6",
        "#bcf60c", "#fabebe"
    ]

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        comm_id = partition[node]
        color = colors[comm_id % len(colors)]

        node_color.append(color)

        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=10,
            color=node_color,
            line=dict(width=1)
        )
    )

    # ---------------- FIGURE ----------------
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Research Collaboration Communities",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # # ---------------- COMMUNITY LIST ----------------
    # st.markdown("### 📌 Detected Communities")

    # for comm_id, nodes in communities.items():
    #     st.markdown(f"""
    #     **{community_names[comm_id]}**  
    #     - Size: {len(nodes)} researchers  
    #     - Sample: {', '.join(nodes[:5])}
    #     """)    