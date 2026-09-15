# privaRAG_agent/features.py
import numpy as np

def compute_friend_features(friends_df):
    """
    Compute I/A/C/E features for each friend
    I = interaction freq
    A = affinity proxy (normalized interactions)
    C = contradiction (dummy here)
    E = engagement ratio
    """
    features = {}
    max_inter = friends_df["interactions"].max()
    for _, row in friends_df.iterrows():
        fid = row["friend"]
        inter = row["interactions"]
        I = inter / max_inter
        A = np.tanh(inter / 10.0)  # simple affinity proxy
        C = np.random.rand() * 0.5  # dummy contradiction
        E = inter / (inter + 5)
        features[fid] = np.array([I, A, C, E])
    return features
