# privaRAG_agent/main.py
from privaRAG_agent.config import *
from privaRAG_agent.data_loader import load_data
from privaRAG_agent.pkg_builder import build_pkg
from privaRAG_agent.features import compute_friend_features
from privaRAG_agent.rag_agent import RAGAgent
from privaRAG_agent.evalulation import evaluate
from privaRAG_agent.visualize_pkg import visualize_pkg

def main():
    # Step 1: Load data
    posts, friends = load_data()

    # Step 2: Build PKG
    G = build_pkg(posts, friends)
    print(f"PKG built with {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Save PNG visualization
    visualize_pkg(G, save_path="outputs/pkg_graph.png")

    # Step 3: Compute features
    friend_features = compute_friend_features(friends)
    print("Friend features computed for", len(friend_features), "friends")

    # Step 4: RAG Agent for explanations
    rag = RAGAgent()
    rag.build_index(posts["content"].tolist())
    explanation = rag.explain("I am going to a political rally")
    print("Generated explanation:", explanation)

    # Dummy evaluation
    y_true = posts["label"].values
    y_pred = (posts["label"].values > 0.5).astype(int)  # trivial baseline
    metrics = evaluate(y_true, y_pred)
    print("Evaluation:", metrics)

if __name__ == "__main__":
    main()
