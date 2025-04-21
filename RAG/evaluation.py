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

model_path = "RAG/node2vec_model.pt"

# Load from saved .pt
model = None
with torch.serialization.safe_globals([CustomWord2Vec]):
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
word2vec = CustomWord2Vec(model.vocab, model.W1, model.word2idx, model.idx2word)

# === Step 1: Load the CSV ===
df = pd.read_csv("RAG/graph_db_with_embeddings.csv")  # Use forward slash for paths

# Separate nodes and edges
node_df = df.iloc[:33302]  # First 33302 rows are nodes
edge_df = df.iloc[33302:]  # From 33303 onwards are edges

# Create a mapping from node ID to name
id_to_name = dict(zip(node_df['_id'], node_df['name']))

# Create graph
G = nx.Graph()

# Add nodes with names and types
for _, row in node_df.iterrows():
    G.add_node(row['_id'], name=row['name'], type=row['node_Type'])

# Add edges
for _, row in edge_df.iterrows():
    G.add_edge(row['_start'], row['_end'], relation=row['_type'])

# === Step 2: Generate 10 Positive Pairs (by name) ===
positive_pairs = []
seen_edges = set()

while len(positive_pairs) < 10:
    edge = edge_df.sample(n=1).iloc[0]
    pair_ids = (edge['_start'], edge['_end'])
    if pair_ids not in seen_edges and (pair_ids[1], pair_ids[0]) not in seen_edges:
        seen_edges.add(pair_ids)
        name_a = id_to_name.get(pair_ids[0])
        name_b = id_to_name.get(pair_ids[1])
        if name_a and name_b:
            positive_pairs.append((name_a, name_b))

# === Step 3: Generate Random Walks to Guide Negative Pair Selection ===
def generate_walk(graph, start_node, length=15):
    walk = [start_node]
    while len(walk) < length:
        current = walk[-1]
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    return walk

# Precompute walks
walks = {}
for node in G.nodes():
    walks[node] = generate_walk(G, node, length=15)

# === Step 4: Generate 10 Negative Pairs (by name) ===
all_nodes = list(G.nodes)
negative_pairs = set()

while len(negative_pairs) < 10:
    node_a, node_b = random.sample(all_nodes, 2)

    if not G.has_edge(node_a, node_b):
        walk_a = walks.get(node_a, [])
        walk_b = walks.get(node_b, [])

        if node_b not in walk_a and node_a not in walk_b:
            name_a = id_to_name.get(node_a)
            name_b = id_to_name.get(node_b)
            if name_a and name_b and (name_a, name_b) not in negative_pairs and (name_b, name_a) not in negative_pairs:
                negative_pairs.add((name_a, name_b))

negative_pairs = list(negative_pairs)

# === Step 5: Display or Save ===
print(" Positive Pairs:")
for a, b in positive_pairs:
    print(f"{a} <-> {b}")

print("\n Negative Pairs:")
for a, b in negative_pairs:
    print(f"{a} !-> {b}")

def compute_similarity(pair):
    a, b = pair
    if a not in word2vec.word2idx or b not in word2vec.word2idx:
        return None
    vec_a = word2vec.wv(a)
    vec_b = word2vec.wv(b)
    sim = cosine_similarity([vec_a], [vec_b])[0][0]
    return sim

# Evaluate similarities
positive_scores = [compute_similarity(pair) for pair in positive_pairs]
negative_scores = [compute_similarity(pair) for pair in negative_pairs]

# Remove None values in case of OOV words
positive_scores = [s for s in positive_scores if s is not None]
negative_scores = [s for s in negative_scores if s is not None]

print("\nAverage similarity of related pairs: ", np.mean(positive_scores))
print("Average similarity of unrelated pairs: ", np.mean(negative_scores))

def plot_tsne(word2vec, words):
    # Filter words with embeddings
    valid_words = [word for word in words if word in word2vec.word2idx]
    vecs = np.array([word2vec.wv(word) for word in valid_words])  # Convert to numpy array

    if len(vecs) < 2:
        print("Not enough valid words for t-SNE.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vecs) - 1))  # Adjust perplexity if needed
    reduced = tsne.fit_transform(vecs)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(valid_words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=12)
    plt.title("t-SNE Visualization of Node Embeddings")
    plt.show()

# === Prepare Words for Plotting ===
words_to_plot = list(set([w for pair in positive_pairs + negative_pairs for w in pair]))  # Flatten pairs
plot_tsne(word2vec, words_to_plot)