import random
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from neo4j import GraphDatabase

# === Neo4j Config ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"  # Replace with your actual password

# === Step 1: Load Graph from Neo4j ===
def load_graph_from_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    graph = nx.Graph()

    with driver.session() as session:
        # We consider all relationships as undirected
        result = session.run("""
            MATCH (a)-[r]-(b)
            WHERE r:HAS_SYMPTOM_OF OR r:TREATS
            RETURN a.name AS src, b.name AS tgt
        """)
        for record in result:
            src = record["src"]
            tgt = record["tgt"]
            if src and tgt:
                graph.add_edge(src, tgt)

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
            walk = biased_random_walk(graph, node, walk_length, p, q)
            walks.append(walk)
    return walks

# === Step 4: Train Word2Vec on Walks ===
def train_node2vec_embeddings(walks, vector_size=128, window=5, min_count=1, sg=1, workers=4):
    model = Word2Vec(
        sentences=walks,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers
    )
    return model

# === Step 5: Main Pipeline ===
if __name__ == "__main__":
    print("Loading graph from Neo4j...")
    G = load_graph_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    print("Generating walks (Node2Vec)...")
    walks = generate_walks(G, num_walks=20, walk_length=30, p=1.0, q=1.0)
    print(f"Generated {len(walks)} walks.")

    print("Training Word2Vec model...")
    model = train_node2vec_embeddings(walks)
    model.wv.save_word2vec_format("node2vec_embeddings.emb")
    print("Embeddings saved to node2vec_embeddings.emb")

    # Example usage
    node_name = "Diabetes"
    if node_name in model.wv:
        print(f"Embedding for '{node_name}':\n{model.wv[node_name]}")
    else:
        print(f"Node '{node_name}' not found in vocabulary.")