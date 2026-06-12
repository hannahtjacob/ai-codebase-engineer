# AI Codebase Engineer

LLM-powered codebase intelligence system that parses repositories, builds
vector and dependency-graph indexes, and answers architectural questions with
source citations.

## Docker Compose

Create a local environment file:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to use OpenAI embeddings and answers. Without a
key, the project uses its deterministic test modes.

Build and start FastAPI, Streamlit, and Neo4j:

```bash
docker compose up --build
```

Open:

- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- Neo4j browser: http://localhost:7474

Run the stack in the background:

```bash
docker compose up --build -d
docker compose logs -f backend frontend
```

Stop the containers:

```bash
docker compose down
```

Repository clones, SQLite metadata, and Chroma indexes persist in the host
`data/` directory. Neo4j data and logs persist in the `neo4j_data` and
`neo4j_logs` Docker volumes. To remove containers and Neo4j volumes:

```bash
docker compose down -v
```

## Local Development

Install the package and test dependencies:

```bash
python -m pip install -e ".[test]"
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

Start the frontend in another terminal:

```bash
streamlit run frontend/streamlit_app.py
```
