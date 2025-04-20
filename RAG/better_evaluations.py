import pandas as pd
import networkx as nx
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomWord2Vec:
    def __init__(self, vocab, W1, word2idx, idx2word):
        self.vocab = vocab
        self.W1 = W1.cpu()
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def wv(self, word):
        return self.W1[self.word2idx[word]].cpu().numpy()

model_path = "RAG/node2vec_model_2.pt"

# Load from saved .pt
model = None
with torch.serialization.safe_globals([CustomWord2Vec]):
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
word2vec = CustomWord2Vec(model.vocab, model.W1, model.word2idx, model.idx2word)

# === Step 1: Load the CSV ===
df = pd.read_csv("RAG/graph_db_with_embeddings_2.csv")

# Separate nodes and edges
node_df = df.iloc[:33302]
edge_df = df.iloc[33302:]

# Create a mapping from node ID to name
id_to_name = dict(zip(node_df['_id'], node_df['name']))

# Create graph
G = nx.Graph()

for _, row in node_df.iterrows():
    G.add_node(row['_id'], name=row['name'], type=row['node_Type'])

for _, row in edge_df.iterrows():
    G.add_edge(row['_start'], row['_end'], relation=row['_type'])

# === Utility Functions ===
def generate_walk(graph, start_node, length=15):
    walk = [start_node]
    while len(walk) < length:
        current = walk[-1]
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    return walk

def compute_similarity(pair):
    a, b = pair
    if a not in word2vec.word2idx or b not in word2vec.word2idx:
        return None
    vec_a = word2vec.wv(a)
    vec_b = word2vec.wv(b)
    sim = cosine_similarity([vec_a], [vec_b])[0][0]
    return sim

def generate_positive_pairs(edge_df, id_to_name, k=10):
    positive_pairs = []
    seen_edges = set()

    while len(positive_pairs) < k:
        edge = edge_df.sample(n=1).iloc[0]
        pair_ids = (edge['_start'], edge['_end'])
        if pair_ids not in seen_edges and (pair_ids[1], pair_ids[0]) not in seen_edges:
            seen_edges.add(pair_ids)
            name_a = id_to_name.get(pair_ids[0])
            name_b = id_to_name.get(pair_ids[1])
            if name_a and name_b:
                positive_pairs.append((name_a, name_b))
    return positive_pairs

def generate_negative_pairs(G, walks, id_to_name, k=10):
    all_nodes = list(G.nodes)
    negative_pairs = set()

    while len(negative_pairs) < k:
        node_a, node_b = random.sample(all_nodes, 2)

        if not G.has_edge(node_a, node_b):
            walk_a = walks.get(node_a, [])
            walk_b = walks.get(node_b, [])

            if node_b not in walk_a and node_a not in walk_b:
                name_a = id_to_name.get(node_a)
                name_b = id_to_name.get(node_b)
                if name_a and name_b and (name_a, name_b) not in negative_pairs and (name_b, name_a) not in negative_pairs:
                    negative_pairs.add((name_a, name_b))
    return list(negative_pairs)

# Precompute random walks for all nodes
walks = {node: generate_walk(G, node, length=15) for node in G.nodes()}

# === Run 10 Iterations ===
all_pos_means = []
all_neg_means = []

for i in range(20):
    print(f"\nIteration {i+1}:")

    pos_pairs = generate_positive_pairs(edge_df, id_to_name, k=100)
    neg_pairs = generate_negative_pairs(G, walks, id_to_name, k=100)

    print("Positive Pairs:")
    for a, b in pos_pairs:
        print(f"{a} <-> {b}")
    
    print("Negative Pairs:")
    for a, b in neg_pairs:
        print(f"{a} !-> {b}")

    pos_scores = [compute_similarity(pair) for pair in pos_pairs]
    neg_scores = [compute_similarity(pair) for pair in neg_pairs]

    # Remove None values in case of OOV words
    pos_scores = [s for s in pos_scores if s is not None]
    neg_scores = [s for s in neg_scores if s is not None]

    pos_mean = np.mean(pos_scores) if pos_scores else 0
    neg_mean = np.mean(neg_scores) if neg_scores else 0

    all_pos_means.append(pos_mean)
    all_neg_means.append(neg_mean)

    print(f"Mean Cosine Similarity (Positive): {pos_mean:.4f}")
    print(f"Mean Cosine Similarity (Negative): {neg_mean:.4f}")

# === Final Results ===
print("\nFinal Results over 20 Iterations:")
print("Average Cosine Similarity (Positive):", np.mean(all_pos_means))
print("Average Cosine Similarity (Negative):", np.mean(all_neg_means))

# === Optional: Plot t-SNE ===
def plot_tsne(word2vec, words):
    valid_words = [word for word in words if word in word2vec.word2idx]
    vecs = np.array([word2vec.wv(word) for word in valid_words])

    if len(vecs) < 2:
        print("Not enough valid words for t-SNE.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vecs) - 1))
    reduced = tsne.fit_transform(vecs)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(valid_words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=12)
    plt.title("t-SNE Visualization of Node Embeddings")
    plt.show()

# Combine all words used in final round
words_to_plot = list(set([w for pair in pos_pairs + neg_pairs for w in pair]))
plot_tsne(word2vec, words_to_plot)