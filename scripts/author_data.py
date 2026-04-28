import pandas as pd
from collections import defaultdict

# Load dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Maps
author_papers_map = defaultdict(list)
paper_authors_map = defaultdict(list)

# Build mappings
for _, row in df.iterrows():
    author = row["author"]
    title = row["title"]

    author_papers_map[author].append(title)
    paper_authors_map[title].append(author)

# Build collaboration map
author_collab_map = defaultdict(list)

for title, authors_list in paper_authors_map.items():
    if len(authors_list) > 1:
        for author in authors_list:
            author_collab_map[author].append(title)


# -----------------------------
# Functions to access data
# -----------------------------
def get_author_papers(author, top_k=3):
    return author_papers_map.get(author, [])[:top_k]


def get_collab_papers(author, top_k=3):
    return author_collab_map.get(author, [])[:top_k]

def get_full_author_data(author, score=None):
    return {
        "author": author,
        "own_papers": author_papers_map.get(author, []),
        "collab_papers": author_collab_map.get(author, []),
        "score": score if score is not None else 0
    }