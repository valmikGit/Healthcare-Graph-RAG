# Healthcare-Graph-RAG

To run the Neo4j Docker image and use our neo4j/data as the /data of the Docker container, run this command:
```bash
docker run --name neo4j -d -p 7474:7474 -p 7687:7687 \
-v /home/ketan/neo4j/neo4j/data:/data \
-v /home/ketan/neo4j/neo4j/logs:/logs \
-e NEO4J_AUTH=none neo4j
```