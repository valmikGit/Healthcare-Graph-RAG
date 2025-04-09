import random
import numpy as np
import networkx as nx
from neo4j import GraphDatabase
from collections import Counter

# === Neo4j Config ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"  # Replace with actual password

# === Step 1: Load Graph from Neo4j as Undirected ===
def load_graph_from_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    graph = nx.Graph()
    with driver.session() as session:
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE r:HAS_SYMPTOM_OF OR r:TREATS
            RETURN a.name AS src, b.name AS tgt
            UNION
            MATCH (a)<-[r]-(b)
            WHERE r:HAS_SYMPTOM_OF OR r:TREATS
            RETURN a.name AS src, b.name AS tgt
        """)
        for record in result:
            src, tgt = record["src"], record["tgt"]
            if src and tgt:
                graph.add_edge(src.lower(), tgt.lower())  # Lowercase consistency
    driver.close()
    return graph

# === Step 2: Biased Random Walk ===
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
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            next_node = np.random.choice(neighbors, p=probabilities)
        walk.append(next_node)
    return walk

# === Step 3: Generate Walks ===
def generate_walks(graph, num_walks, walk_length, p=1.0, q=1.0):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(biased_random_walk(graph, node, walk_length, p, q))
    return walks

# === Step 4: Train Skip-Gram Word2Vec ===
class CustomWord2Vec:
    def __init__(self, vocab, W1, word2idx, idx2word):
        self.vocab = vocab
        self.W1 = W1
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def wv(self, word):
        return self.W1[self.word2idx[word]]

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
        return np.array(data)

    training_data = generate_training_data(filtered_sentences, window)

    W1 = np.random.uniform(-0.8, 0.8, (vocab_size, vector_size))
    W2 = np.random.uniform(-0.8, 0.8, (vector_size, vocab_size))

    def forward(center_idx):
        h = W1[center_idx]
        u = np.dot(h, W2)
        y_hat = np.exp(u - np.max(u))
        y_hat /= np.sum(y_hat)
        return y_hat, h

    def backward(y_hat, h, target_idx, center_idx):
        nonlocal W1, W2
        error = y_hat
        error[target_idx] -= 1
        dW2 = np.outer(h, error)
        dW1 = np.dot(W2, error)
        W2 -= learning_rate * dW2
        W1[center_idx] -= learning_rate * dW1

    for epoch in range(epochs):
        np.random.shuffle(training_data)
        loss = 0
        for center_idx, target_idx in training_data:
            y_hat, h = forward(center_idx)
            backward(y_hat, h, target_idx, center_idx)
            loss += -np.log(y_hat[target_idx] + 1e-9)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return CustomWord2Vec(vocab, W1, word2idx, idx2word)

# === Step 5: Save Embeddings Back to Neo4j ===
def save_embeddings_to_neo4j(uri, user, password, model):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for word in model.vocab:
            emb = model.wv(word).tolist()
            session.run("""
                MATCH (n)
                WHERE toLower(n.name) = $name
                SET n.vector_embedding = $embedding
            """, name=word, embedding=emb)
    driver.close()
    print("✅ Embeddings saved to Neo4j.")

# === Step 6: Main Execution ===
if __name__ == "__main__":
    print("🔗 Loading graph from Neo4j...")
    G = load_graph_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    print(f"✅ Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    print("🚶 Generating walks...")
    walks = generate_walks(G, num_walks=20, walk_length=30)

    print("🧠 Training embeddings from scratch...")
    model = train_node2vec_embeddings(walks, vector_size=128, epochs=50)

    print("📤 Uploading embeddings to Neo4j...")
    save_embeddings_to_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, model)

    print("🎉 Done!")