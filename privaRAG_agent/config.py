# privaRAG_agent/config.py
DATA_PATH = "data/facebook_dataset.csv"
FRIEND_PATH = "data/friendships.csv"
PKG_PATH = "data/pkg.pkl"

# Hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 16

# Federated Learning
NUM_CLIENTS = 10
ROUNDS = 5
DP_NOISE = 1.2
CLIP_NORM = 1.0

# Retrieval
TOP_K = 5
