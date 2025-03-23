# Healthcare-Graph-RAG
Course project in Natural Language Processing

```bash
git clone https://github.com/doccano/doccano.git
```
```bash
cd doccano/
 ```
 ```bash
cd docker/
```
```bash
ADMIN_USERNAME=valmik_admin_username ADMIN_PASSWORD=valmik_admin_password ADMIN_EMAIL=valmik0000000@gmail.com POSTGRES_USER=valmik_postgres_username POSTGRES_PASSWORD=valmik_postgres_password POSTGRES_DB=valmik_postgres_db RABBITMQ_DEFAULT_USER=valmik_rabbit_mq_username RABBITMQ_DEFAULT_PASS=valmik_rabbit_mq_password docker compose -f docker-compose.prod.yml up --build
```
```bash
docker compose -f docker-compose.prod.yml down
```