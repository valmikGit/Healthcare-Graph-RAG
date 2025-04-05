from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from neo4j import GraphDatabase
import pandas as pd

# Load pretrained NER pipeline
model_name = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=False)

# Neo4j credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "valmik_neo4j"  # Change this

# Initialize Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Function to process a single sentence using NER
def extract_entities(text):
    ner_results = ner_pipeline(text)
    entities = []
    current_entity = ""
    current_label = ""
    current_start = None
    current_end = None

    for item in ner_results:
        word = item["word"]
        label = item["entity"]
        start = item["start"]
        end = item["end"]

        if "-" in label:
            _, entity_type = label.split("-")
        else:
            entity_type = label

        if entity_type == "O":
            if current_entity:
                entities.append((current_entity, current_label, current_start, current_end))
                current_entity = ""
                current_label = ""
            continue

        if current_entity:
            if (start == current_end) and (entity_type == current_label):
                current_entity += word[2:] if word.startswith("##") else " " + word
                current_end = end
            else:
                entities.append((current_entity, current_label, current_start, current_end))
                current_entity = word[2:] if word.startswith("##") else word
                current_label = entity_type
                current_start = start
                current_end = end
        else:
            current_entity = word[2:] if word.startswith("##") else word
            current_label = entity_type
            current_start = start
            current_end = end

    if current_entity:
        entities.append((current_entity, current_label, current_start, current_end))

    final_entities = []
    for ent in entities:
        text_segment = text[ent[2]:ent[3]]
        final_entity = text_segment if ent[0].lower() != text_segment.lower() else ent[0]
        final_entities.append((final_entity, ent[1]))

    # Merge Detailed_description followed by Sign_symptom
    merged_entities = []
    i = 0
    while i < len(final_entities):
        if i < len(final_entities) - 1:
            curr_ent, next_ent = final_entities[i], final_entities[i+1]
            if curr_ent[1] == "Detailed_description" and next_ent[1] == "Sign_symptom":
                merged_text = f"{curr_ent[0]} {next_ent[0]}"
                merged_entities.append((merged_text, "Sign_symptom"))
                i += 2
                continue
        merged_entities.append(final_entities[i])
        i += 1

    seen = set()
    unique_entities = []
    for ent in merged_entities:
        if ent[1] != "O":
            lowercased = ent[0].strip().lower()
            if lowercased not in seen:
                seen.add(lowercased)
                unique_entities.append((lowercased, ent[1]))

    return unique_entities

# Function to push entities to Neo4j
def add_to_neo4j(tx, disease, symptoms, treatments):
    # Create or reuse disease node
    disease_query = """
    MERGE (d:Disease {name: $name, node_Type: 'Disease'})
    ON CREATE SET d.vector_embedding = $embedding
    """
    tx.run(disease_query, name=disease, embedding=[])

    # Create or reuse symptom nodes and relationships
    for symptom in symptoms:
        symptom_query = """
        MERGE (s:Symptom {name: $name, node_Type: 'Symptom'})
        ON CREATE SET s.vector_embedding = $embedding
        WITH s
        MATCH (d:Disease {name: $disease, node_Type: 'Disease'})
        MERGE (d)-[:HAS_SYMPTOM]->(s)
        """
        tx.run(symptom_query, name=symptom, embedding=[], disease=disease)

    # Create or reuse treatment nodes and relationships
    for treatment in treatments:
        treatment_query = """
        MERGE (t:Treatment {name: $name, node_Type: 'Treatment'})
        ON CREATE SET t.vector_embedding = $embedding
        WITH t
        MATCH (d:Disease {name: $disease, node_Type: 'Disease'})
        MERGE (t)-[:TREATS]->(d)
        """
        tx.run(treatment_query, name=treatment, embedding=[], disease=disease)

# Process dataset (assume CSV with one sentence per row)
df = pd.read_csv("split_sentences.csv")  # Must have a 'text' column
df = df.dropna(subset=["answer"])

with driver.session() as session:
    for idx, row in df.iterrows():
        sentence = row["answer"]
        entities = extract_entities(sentence)

        diseases = [e[0].strip().lower() for e in entities if e[1] == "Disease_disorder"]
        symptoms = [e[0].strip().lower() for e in entities if e[1] == "Sign_symptom"]
        treatments = [e[0].strip().lower() for e in entities if e[1] == "Medication"]

        for disease in diseases:
            session.execute_write(add_to_neo4j, disease, symptoms, treatments)

driver.close()