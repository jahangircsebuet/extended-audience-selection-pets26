# privaRAG_agent/data_loader.py
import json
import pandas as pd

def load_data(json_path="data/dummy_data.json"):
    """Load posts and friends from JSON"""
    with open(json_path, "r") as f:
        data = json.load(f)

    posts = pd.DataFrame(data["posts"])
    friends = pd.DataFrame(data["friends"])
    return posts, friends
