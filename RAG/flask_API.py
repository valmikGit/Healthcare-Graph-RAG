from flask import Flask, request, jsonify
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer as QATokenizer
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

# ======================
#      Configuration
# ======================
NER_MODEL_NAME = "Helios9/BioMed_NER"
QA_MODEL_NAME = "deepset/bert-base-cased-squad2"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "valmik_neo4j"
EMBEDDING_MODEL_PATH = "node2vec_model.pt"
MAX_CONTEXT_LENGTH = 512

# ======================
#  Custom Word2Vec Class 
# ======================
class CustomWord2Vec:
    def __init__(self, vocab, W1, word2idx, idx2word):
        self.vocab = vocab
        self.W1 = W1.cpu()
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def wv(self, word):
        return self.W1[self.word2idx[word]].cpu().numpy()

# ======================
#      NER Component
# ======================
class BiomedicalNER:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
        self.model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
    
    def extract_entities(self, text):
        ner_results = self.pipeline(text)
        entities = []
        seen = set()

        for entity in ner_results:
            if entity["entity_group"] == "O":
                continue
            text_segment = text[entity["start"]:entity["end"]]
            text_segment_clean = text_segment.strip().lower()
            if text_segment_clean not in seen:
                seen.add(text_segment_clean)
                entities.append((text_segment_clean, entity["entity_group"]))
        
        # Merge Detailed_description + Sign_symptom
        merged_entities = []
        i = 0
        while i < len(entities):
            if i < len(entities) - 1:
                current_ent, next_ent = entities[i], entities[i+1]
                if current_ent[1] == "Detailed_description" and next_ent[1] == "Sign_symptom":
                    merged_text = f"{current_ent[0]} {next_ent[0]}"
                    merged_entities.append((merged_text, "Sign_symptom"))
                    i += 2
                    continue
            merged_entities.append(entities[i])
            i += 1

        return [ent[0] for ent in merged_entities]

# ======================
#  Embedding & Similarity
# ======================
class NodeEmbeddings:
    def __init__(self, model_path):
        with torch.serialization.safe_globals([CustomWord2Vec]):
            self.model = torch.load(
                model_path,
                map_location=torch.device('cpu'),
                weights_only=True
            )
        self.vocab = self.model.vocab
        self.word2idx = self.model.word2idx
        self.embeddings = self.model.W1.cpu().numpy()
        
    def get_embedding(self, text):
        words = text.lower().split()
        valid_embs = []
        for word in words:
            if word in self.word2idx:
                valid_embs.append(self.embeddings[self.word2idx[word]])
        return np.mean(valid_embs, axis=0).tolist() if valid_embs else None

    def top_similar_nodes(self, embedding, top_k=5):
        if embedding is None:
            return []
        similarities = cosine_similarity([embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.vocab[i], float(similarities[i])) for i in top_indices]

# ======================
#    Neo4j Connector
# ======================
class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def get_relationships(self, node_name):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                WHERE toLower(a.name) = toLower($name)
                RETURN a.name AS src, type(r) AS rel, b.name AS tgt
                UNION
                MATCH (a)<-[r]-(b)
                WHERE toLower(a.name) = toLower($name)
                RETURN a.name AS src, type(r) AS rel, b.name AS tgt
            """, name=node_name)
            return [f"{record['src']} {record['rel'].replace('_',' ').lower()} {record['tgt']}".lower() for record in result]

# ======================
#       QA Component
# ======================
class BiomedicalQA:
    def __init__(self):
        self.qa_tokenizer = QATokenizer.from_pretrained(QA_MODEL_NAME)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
        
    def answer_question(self, context, question):
        if not context.strip():
            return "No relevant information found in the knowledge base"
            
        try:
            inputs = self.qa_tokenizer(
                question,
                context,
                add_special_tokens=True,
                max_length=MAX_CONTEXT_LENGTH,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt"
            )
            outputs = self.qa_model(**inputs)
            start_logits = outputs.start_logits.detach().numpy().flatten()
            end_logits = outputs.end_logits.detach().numpy().flatten()
            token_type_ids = inputs.token_type_ids.numpy().flatten()
            context_indices = np.where(token_type_ids == 1)[0]
            if len(context_indices) == 0:
                return "No relevant context available for answering"
            context_start = context_indices[0]
            context_end = context_indices[-1]
            start_logits[:context_start] = -np.inf
            end_logits[:context_start] = -np.inf
            start_logits[context_end+1:] = -np.inf
            end_logits[context_end+1:] = -np.inf
            start_idx = np.argmax(start_logits)
            end_idx = np.argmax(end_logits)
            if end_idx < start_idx or start_idx == 0:
                return "No clear answer found in the knowledge base"
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
            answer = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            return answer if answer else "No specific answer found"
        except Exception as e:
            return f"Error processing answer: {str(e)}"

# ======================
#      Flask App
# ======================
app = Flask(__name__)
ner = BiomedicalNER()
node_embeddings = NodeEmbeddings(EMBEDDING_MODEL_PATH)
neo4j_conn = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
qa = BiomedicalQA()

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    user_query = data.get("question", "").strip()
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    response = {
        "entities": [],
        "embeddings": {},
        "retrieved_sentences": [],
        "context": "",
        "answer": ""
    }

    entities = ner.extract_entities(user_query)
    response["entities"] = entities

    context_chunks = []
    for entity in entities:
        emb = node_embeddings.get_embedding(entity)
        response["embeddings"][entity] = emb
        if emb is None:
            continue
        similar_nodes = node_embeddings.top_similar_nodes(emb)
        for node_name, _ in similar_nodes:
            relationships = neo4j_conn.get_relationships(node_name)
            context_chunks.extend(relationships)

    unique_chunks = list(set(context_chunks))
    response["retrieved_sentences"] = unique_chunks
    final_context = " ".join(sorted(unique_chunks, key=len))[:3000]
    response["context"] = final_context
    response["answer"] = qa.answer_question(final_context, user_query)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)