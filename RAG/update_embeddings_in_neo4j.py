import pandas as pd
import ast
from neo4j import GraphDatabase

# === Neo4j credentials ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "valmik_neo4j"

# === Load CSV ===
csv_path = "RAG\graph_db_with_embeddings.csv"
df = pd.read_csv(csv_path, low_memory=False)

# Drop rows that are not nodes (after row 33302)
df = df.iloc[:33302]

# === Clean and convert vector_embedding ===
def parse_embedding(embedding_str):
    if pd.isna(embedding_str) or embedding_str.strip() == "":
        return None
    try:
        return list(map(float, ast.literal_eval(embedding_str)))
    except Exception:
        return None

df['vector_embedding'] = df['vector_embedding'].apply(parse_embedding)

# === Connect to Neo4j ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === Update vector embeddings ===
def update_embedding(tx, name, node_type, label, embedding):
    cypher = f"""
    MATCH (n:{label} {{name: $name, node_Type: $node_type}})
    SET n.vector_embedding = $embedding
    """
    tx.run(cypher, name=name, node_type=node_type, embedding=embedding)

with driver.session() as session:
    for idx, row in df.iterrows():
        name = row['name']
        node_type = row['node_Type']
        label = row['_labels'].replace(":", "")  # Remove ':' from label
        embedding = row['vector_embedding']
        
        try:
            session.execute_write(update_embedding, name, node_type, label, embedding)
            print(f"Updated node {name} [{label}] with embedding.")
        except Exception as e:
            print(f"Failed to update node {name} [{label}]: {e}")

driver.close()
print("✅ All embeddings updated.")