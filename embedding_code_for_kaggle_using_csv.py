import random
import torch
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

# === Device Configuration ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === CSV Config ===
CSV_PATH = "graph_db.csv"
NODE_ROWS = 33302  # First 33302 rows are nodes

# === Graph Construction ===
def load_graph_from_csv(csv_path):
    """Load graph from CSV file, ignoring isolated nodes"""
    df = pd.read_csv(csv_path)
    
    # Split into nodes and relationships
    nodes_df = df.head(NODE_ROWS)
    rels_df = df.tail(len(df) - NODE_ROWS)
    
    # Create ID to name mapping
    id_to_name = {row['_id']: row['name'].lower() for _, row in nodes_df.iterrows()}
    
    # Build graph with NetworkX
    G = nx.Graph()
    
    # Add nodes with connections
    connected_nodes = set()
    for _, rel in rels_df.iterrows():
        if pd.notna(rel['_start']) and pd.notna(rel['_end']):
            src = id_to_name.get(rel['_start'], '').lower()
            tgt = id_to_name.get(rel['_end'], '').lower()
            if src and tgt:
                G.add_edge(src, tgt)
                connected_nodes.add(src)
                connected_nodes.add(tgt)
    
    # Remove isolated nodes
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
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(biased_random_walk(graph, node, walk_length, p, q))
    return walks

# === GPU-accelerated Word2Vec ===
class CustomWord2Vec:
    def __init__(self, vocab, W1, word2idx, idx2word):
        self.vocab = vocab
        self.W1 = W1.to(device)
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def wv(self, word):
        return self.W1[self.word2idx[word]].cpu().numpy()

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

    # Initialize weights on GPU
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
        indices = torch.randperm(len(training_data), device=device)
        training_data = training_data[indices]
        loss = 0
        
        for center_idx, target_idx in training_data:
            y_hat, h = forward(center_idx)
            backward(y_hat, h, target_idx, center_idx)
            loss += -torch.log(y_hat[target_idx] + 1e-9)
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()/len(training_data):.4f}")

    return CustomWord2Vec(vocab, W1, word2idx, idx2word)

# === Save Embeddings to CSV ===
def save_embeddings_to_csv(original_df, nodes_df, model, id_to_name):
    """Update the original DataFrame with embeddings"""
    # Create reverse mapping from name to ID
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}
    
    # Create embedding dictionary
    embeddings = {name: model.wv(name).tolist() for name in model.vocab}
    
    # Update vector_embedding column
    def update_embedding(row):
        if row['node_Type'] in ['Symptom', 'Disease', 'Treatment']:
            lower_name = row['name'].lower()
            if lower_name in embeddings:
                return embeddings[lower_name]
        return []
    
    original_df['vector_embedding'] = original_df.apply(update_embedding, axis=1)
    
    # Save back to CSV
    original_df.to_csv(CSV_PATH, index=False)
    print("✅ Embeddings saved to CSV.")

# === Main Execution ===
if __name__ == "__main__":
    print("🔗 Loading graph from CSV...")
    G, nodes_df, id_to_name = load_graph_from_csv(CSV_PATH)
    print(f"✅ Loaded graph with {len(G.nodes())} connected nodes and {len(G.edges())} edges.")

    print("🚶 Generating walks...")
    walks = generate_walks(G, num_walks=20, walk_length=30)

    print("🧠 Training embeddings on GPU...")
    model = train_node2vec_embeddings(walks, vector_size=128, epochs=50)

    print("📤 Updating CSV with embeddings...")
    original_df = pd.read_csv(CSV_PATH)
    save_embeddings_to_csv(original_df, nodes_df, model, id_to_name)

    print("🎉 Done!")