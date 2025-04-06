import random
import numpy as np
import networkx as nx
from collections import defaultdict
from gensim.models import Word2Vec

# Example Graph (Healthcare Knowledge Graph)
graph = {
    'Diabetes': ['Insulin', 'Metformin', 'Blurred Vision'],
    'Insulin': ['Diabetes'],
    'Metformin': ['Diabetes'],
    'Blurred Vision': ['Diabetes', 'Cataract'],
    'Cataract': ['Blurred Vision']
}

# Convert to NetworkX Graph
G = nx.Graph()
for src, targets in graph.items():
    for tgt in targets:
        G.add_edge(src, tgt)

def biased_random_walk(graph, start_node, walk_length, p=1.0, q=1.0):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph[cur])
        
        if len(neighbors) == 0:
            break  # End walk if no neighbors
        
        if len(walk) == 1:
            # Choose randomly in first step
            next_node = random.choice(neighbors)
        else:
            prev = walk[-2]  # Previous node in walk
            prob = []
            for neighbor in neighbors:
                if neighbor == prev:
                    prob.append(1 / p)  # Returning to previous node
                elif neighbor in graph[cur]:  
                    prob.append(1)  # Nearby neighbors (default probability)
                else:
                    prob.append(1 / q)  # Exploring further nodes
            
            prob = np.array(prob) / sum(prob)  # Normalize probabilities
            next_node = np.random.choice(neighbors, p=prob)
        
        walk.append(next_node)
    
    return walk

def generate_walks(graph, num_walks, walk_length, p=1.0, q=1.0):
    walks = []
    nodes = list(graph.keys())
    
    for _ in range(num_walks):
        random.shuffle(nodes)  # Shuffle to avoid bias
        for node in nodes:
            walks.append(biased_random_walk(graph, node, walk_length, p, q))
    
    return walks

# Generate walks
walks = generate_walks(G, num_walks=10, walk_length=10, p=1.0, q=1.0)
print(walks)  # Example: [['Diabetes', 'Insulin', 'Diabetes', 'Metformin', ...], ...]

# Train Word2Vec Model
model = Word2Vec(sentences=walks, vector_size=128, window=5, min_count=1, sg=1, workers=4)

# Save and Load Embeddings
model.wv.save_word2vec_format("node2vec_custom.emb")

# Retrieve embedding of 'Diabetes'
print(model.wv['Diabetes'])