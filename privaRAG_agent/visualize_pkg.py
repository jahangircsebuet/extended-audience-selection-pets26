# privaRAG_agent/visualize_pkg.py
import matplotlib.pyplot as plt
import networkx as nx
import os

def visualize_pkg(G, save_path="pkg_visualization.png"):
    """
    Visualize and save the Privacy Knowledge Graph (PKG) as a PNG file.
    
    Args:
        G (networkx.Graph): The PKG graph object
        save_path (str): Path to save the PNG image
    """
    # Ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    
    # Assign colors based on node type
    color_map = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "user":
            color_map.append("skyblue")
        elif data.get("type") == "post":
            color_map.append("lightgreen")
        elif data.get("type") == "friend":
            color_map.append("orange")
        else:
            color_map.append("lightgrey")
    
    # Layout for visualization
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes, edges, and labels
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=color_map,
        node_size=800,
        font_size=8,
        font_weight="bold",
        edge_color="gray"
    )
    
    plt.title("Privacy Knowledge Graph (PKG)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)   # Save as PNG
    plt.close()
    print(f"[INFO] PKG visualization saved to {save_path}")
