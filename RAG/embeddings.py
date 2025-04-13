import random
import torch
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import os
import time

# === Device Configuration ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === File Path Config ===
CSV_PATH = "graph_db.csv"
MODEL_SAVE_PATH = "node2vec_model.pt"
NODE_ROWS = 33302

# === Utility ===
def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

# === Graph Construction ===
def load_graph_from_csv(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    nodes_df = df.head(NODE_ROWS)
    rels_df = df.tail(len(df) - NODE_ROWS)
    id_to_name = {}
    for _, row in nodes_df.iterrows():
        try:
            row_id = str(row['_id']) if pd.notna(row['_id']) else None
            name = str(row['name']).lower() if pd.notna(row['name']) else ''
            if row_id and name:
                id_to_name[row_id] = name
        except:
            continue

    G = nx.Graph()
    for _, rel in rels_df.iterrows():
        try:
            start = str(rel['_start']) if pd.notna(rel['_start']) else None
            end = str(rel['_end']) if pd.notna(rel['_end']) else None
            if start and end:
                src = id_to_name.get(start, '').lower()
                tgt = id_to_name.get(end, '').lower()
                if src and tgt:
                    G.add_edge(src, tgt)
        except:
            continue

    G.remove_nodes_from(list(nx.isolates(G)))
    return G, nodes_df, id_to_name

# === Biased Random Walk ===
def biased_random_walk(graph, start_node, walk_length, p=1.0, q=1.0):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if not neighbors:
            break
        if len(walk) == 1:
            next_node = random.choice(neighbors)
        else:
            prev = walk[-2]
            probabilities = []
            for neighbor in neighbors:
                if neighbor == prev:
                    probabilities.append(1 / p)
                elif graph.has_edge(neighbor, prev):
                    probabilities.append(1)
                else:
                    probabilities.append(1 / q)
            probabilities = torch.tensor(probabilities, device=device)
            probabilities /= probabilities.sum()
            next_node = neighbors[torch.multinomial(probabilities, 1).item()]
        walk.append(next_node)
    return walk

# === Generate Walks ===
def generate_walks(graph, num_walks, walk_length, p=1.0, q=1.0):
    walks = []
    nodes = list(graph.nodes())
    total = num_walks * len(nodes)
    start_time = time.time()
    print(f"\n🚶 Generating {total} walks...")

    for i in range(num_walks):
        random.shuffle(nodes)
        for j, node in enumerate(nodes):
            walks.append(biased_random_walk(graph, node, walk_length, p, q))
            count = i * len(nodes) + j + 1
            if count % 1000 == 0 or count == total:
                elapsed = time.time() - start_time
                print(f"[{count}/{total} {format_time(elapsed)}]")

    print(f"✅ Done generating walks in {format_time(time.time() - start_time)}")
    return walks

# === GPU Word2Vec Training ===
class CustomWord2Vec:
    def __init__(self, vocab, W1, word2idx, idx2word):
        self.vocab = vocab
        self.W1 = W1.to(device)
        self.word2idx = word2idx
        self.idx2word = idx2word

    def wv(self, word):
        return self.W1[self.word2idx[word]].cpu().numpy().astype(np.float32)

    def save(self, path):
        torch.save({
            'vocab': self.vocab,
            'W1': self.W1,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        return cls(
            vocab=checkpoint['vocab'],
            W1=checkpoint['W1'],
            word2idx=checkpoint['word2idx'],
            idx2word=checkpoint['idx2word']
        )

# === Train Node2Vec Embeddings ===
def train_node2vec_embeddings(walks, vector_size=128, window=5, min_count=1, epochs=10, learning_rate=0.01):
    sentences = [[word.lower() for word in walk] for walk in walks]
    word_counts = Counter([word for sentence in sentences for word in sentence])
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(vocab)

    filtered_sentences = [[w for w in s if w in word2idx] for s in sentences]

    def generate_training_data(sentences, window_size):
        data = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                context = sentence[max(0, i-window_size):i] + sentence[i+1:i+window_size+1]
                for ctx in context:
                    data.append((word2idx[word], word2idx[ctx]))
        return torch.tensor(data, device=device)

    training_data = generate_training_data(filtered_sentences, window)
    W1 = torch.empty((vocab_size, vector_size), device=device).uniform_(-0.8, 0.8)
    W2 = torch.empty((vector_size, vocab_size), device=device).uniform_(-0.8, 0.8)

    def forward(center_idx):
        h = W1[center_idx]
        u = torch.matmul(h, W2)
        y_hat = torch.exp(u - torch.max(u))
        y_hat /= torch.sum(y_hat)
        return y_hat, h

    def backward(y_hat, h, target_idx, center_idx):
        nonlocal W1, W2
        error = y_hat.clone()
        error[target_idx] -= 1
        dW2 = torch.outer(h, error)
        dW1 = torch.matmul(W2, error)
        W2 -= learning_rate * dW2
        W1[center_idx] -= learning_rate * dW1

    for epoch in range(epochs):
        print(f"\n🧠 Epoch {epoch + 1}/{epochs} starting...")
        start_time = time.time()
        indices = torch.randperm(len(training_data), device=device)
        training_data = training_data[indices]
        loss = 0
        total = len(training_data)

        for i, (center_idx, target_idx) in enumerate(training_data):
            y_hat, h = forward(center_idx)
            backward(y_hat, h, target_idx, center_idx)
            loss += -torch.log(y_hat[target_idx] + 1e-9)

            if (i + 1) % 10000 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                print(f"[{i+1}/{total} {format_time(elapsed)}, Epoch {epoch + 1}/{epochs}]")

        avg_loss = loss.item() / total
        print(f"✅ Epoch {epoch + 1} completed in {format_time(time.time() - start_time)} | Avg Loss: {avg_loss:.4f}")

    return CustomWord2Vec(vocab, W1, word2idx, idx2word)

# === Save Embeddings to CSV ===
def save_embeddings_to_csv(original_df, nodes_df, model, id_to_name):
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}
    embeddings = {name: model.wv(name) for name in model.vocab}

    def update_embedding(row):
        if row['node_Type'] in ['Symptom', 'Disease', 'Treatment']:
            lower_name = row['name'].lower()
            if lower_name in embeddings:
                return list(embeddings[lower_name])
        return None

    original_df['vector_embedding'] = original_df.apply(update_embedding, axis=1)
    original_df.to_csv(CSV_PATH, index=False)
    print("Embeddings saved to CSV.")

# === Main Execution ===
if __name__ == "__main__":
    print("Loading graph from CSV...")
    G, nodes_df, id_to_name = load_graph_from_csv(CSV_PATH)
    print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    walks = generate_walks(G, num_walks=10, walk_length=20)
    model = train_node2vec_embeddings(walks, vector_size=128, epochs=3)

    torch.save(model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    original_df = pd.read_csv(CSV_PATH)
    save_embeddings_to_csv(original_df, nodes_df, model, id_to_name)
    print("✅ All done!")