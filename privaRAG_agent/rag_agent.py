# privaRAG_agent/rag_agent.py
import faiss
import numpy as np
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

class RAGAgent:
    def __init__(self, embedder="sentence-transformers/all-MiniLM-L6-v2"):
        # Sentence embeddings for retrieval
        self.model = SentenceTransformer(embedder)
        self.index = None
        self.posts = []

        # Load LLM with Hugging Face token (fallback to distilgpt2 if token not found)
        model_name = "gpt2"
        if HF_TOKEN is None:
            print("[WARN] Hugging Face token not found in .env — using distilgpt2 (public).")
            model_name = "distilgpt2"

        self.llm = pipeline(
            "text-generation",
            model=model_name,
            use_auth_token=HF_TOKEN if HF_TOKEN else None
        )

    def build_index(self, texts):
        embeddings = self.model.encode(texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        self.posts = texts

    def explain(self, new_post):
        q_emb = self.model.encode([new_post])
        D, I = self.index.search(q_emb, 3)
        retrieved = [self.posts[i] for i in I[0]]
        context = " ".join(retrieved)
        prompt = f"Given the post: {new_post}\nSimilar past posts: {context}\nExplain why audience restrictions are recommended."
        return self.llm(prompt, max_length=80, num_return_sequences=1)[0]["generated_text"]
