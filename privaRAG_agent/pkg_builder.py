# privaRAG_agent/pkg_builder.py
import networkx as nx
import pickle
from privaRAG_agent.config import PKG_PATH

def build_pkg(posts, friends):
    """Construct Privacy Knowledge Graph"""
    G = nx.Graph()
    for _, row in posts.iterrows():
        post_id = f"post_{row['post_id']}"
        G.add_node(post_id, type="post", text=row['content'], label=row['label'])
        G.add_node(row['user'], type="user")
        G.add_edge(row['user'], post_id, relation="authored")

    for _, row in friends.iterrows():
        G.add_node(row['friend'], type="friend")
        G.add_edge(row['user'], row['friend'], weight=row['interactions'], relation="friendship")

    with open(PKG_PATH, "wb") as f:
        pickle.dump(G, f)
    return G
