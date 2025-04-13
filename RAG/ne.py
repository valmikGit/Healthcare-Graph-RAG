# import torch
# import pandas as pd
# import numpy as np
# import os

# # === Config ===
# MODEL_PATH = r"C:\Users\HP\Downloads\RAG\RAG\node2vec_model.pt"
# OUTPUT_CSV = r"C:\Users\HP\Downloads\RAG\RAG\node_embeddings.csv"

# # === CustomWord2Vec Class ===
# class CustomWord2Vec:
#     def __init__(self, vocab, W1, word2idx, idx2word):
#         self.vocab = vocab
#         self.W1 = W1
#         self.word2idx = word2idx
#         self.idx2word = idx2word

#     def wv(self, word):
#         return self.W1[self.word2idx[word]].cpu().numpy().astype(np.float32)

#     @classmethod
#     def load(cls, path):
#         return torch.load(path, map_location='cpu', weights_only=False)

# # === Load the model ===
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# print("📦 Loading model...")
# model = CustomWord2Vec.load(MODEL_PATH)
# print(f"✅ Model loaded with {len(model.vocab)} nodes.")

# # === Extract embeddings and save to CSV ===
# print("💾 Saving embeddings to CSV...")

# data = []
# for word in model.vocab:
#     vec = model.wv(word)
#     data.append([word] + vec.tolist())

# # Create DataFrame
# embedding_df = pd.DataFrame(data)
# embedding_df.columns = ['node'] + [f'dim_{i}' for i in range(embedding_df.shape[1] - 1)]

# # Save
# embedding_df.to_csv(OUTPUT_CSV, index=False)
# print(f"✅ Embeddings saved to {OUTPUT_CSV}")


import torch
import pandas as pd
import numpy as np
import os

# === Paths ===
MODEL_PATH = r"C:\Users\HP\Downloads\RAG\RAG\node2vec_model.pt"
CSV_PATH = r"C:\Users\HP\Downloads\RAG\RAG\graph_db.csv"  # Uploaded file path
OUTPUT_CSV = r"C:\Users\HP\Downloads\RAG\RAG\graph_db_with_embeddings.csv"

# === CustomWord2Vec Class ===
class CustomWord2Vec:
    def __init__(self, vocab, W1, word2idx, idx2word):
        self.vocab = vocab
        self.W1 = W1
        self.word2idx = word2idx
        self.idx2word = idx2word

    def wv(self, word):
        return self.W1[self.word2idx[word]].cpu().numpy().astype(np.float32)

    @classmethod
    def load(cls, path):
        return torch.load(path, map_location='cpu', weights_only=False)

# === Load the model ===
print("📦 Loading model...")
model = CustomWord2Vec.load(MODEL_PATH)
print(f"✅ Model loaded with {len(model.vocab)} nodes.")

# === Load graph_db.csv ===
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"📊 Loaded CSV with shape: {df.shape}")

# === Embed nodes where possible ===
def get_embedding(row):
    if pd.isna(row.get('name')) or pd.isna(row.get('node_Type')):
        return None
    if row['node_Type'] in ['Symptom', 'Disease', 'Treatment']:
        key = str(row['name']).lower()
        if key in model.vocab:
            return model.wv(key).tolist()
    return None

print("🧠 Adding embeddings to rows...")
df['vector_embedding'] = df.apply(get_embedding, axis=1)

# === Save to CSV ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Enriched CSV saved to: {OUTPUT_CSV}")
