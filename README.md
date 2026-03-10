# Hybrid RAG Support API

Backend para responder preguntas sobre manuales de soporte técnico usando Hybrid Retrieval-Augmented Generation.

## Stack

| Componente | Tecnología |
|---|---|
| API | FastAPI |
| Validación | Pydantic v2 |
| Historial | Redis |
| Vector DB | Qdrant |
| LLM / Embeddings | Ollama |
| Retrieval | Vector Search + Keyword Search (RRF) |

## Estructura

```
app/
├── main.py                  # FastAPI app entry point
├── config.py                # Settings desde .env (pydantic-settings)
├── api/
│   └── routes/
│       ├── chat.py          # POST /api/v1/chat
│       └── health.py        # GET  /api/v1/health[/ready]
├── models/
│   └── schemas.py           # Pydantic models (request/response/domain)
├── services/
│   ├── rag_service.py       # Orquestador principal del pipeline RAG
│   ├── retrieval_service.py # Hybrid search con Qdrant + RRF
│   ├── llm_service.py       # Embeddings y generación con Ollama
│   └── redis_memory.py      # Historial de conversaciones en Redis
└── utils/
    └── prompt_builder.py    # Construcción del prompt final
```

## Requisitos previos

- Docker & Docker Compose
- Ollama corriendo localmente con los modelos descargados:
  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text
  ```
- Colección `tech_manuals` ya cargada en Qdrant con índice de texto en el campo `text`

## Inicio rápido

```bash
# 1. Clonar y configurar
cp .env.example .env
# Editar .env según tu entorno

# 2. Levantar servicios con Docker Compose
docker compose up -d

# 3. Verificar que todo esté listo
curl http://localhost:8000/api/v1/health/ready
```

## Uso de la API

### POST /api/v1/chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "user-session-001",
    "question": "¿Cómo reinicio el servidor de base de datos?"
  }'
```

**Response:**
```json
{
  "answer": "Para reiniciar el servidor de base de datos...",
  "sources": ["manual_db_v2.pdf", "guia_operaciones.pdf"]
}
```

### GET /api/v1/health/ready

Verifica conectividad con Redis, Qdrant y Ollama.

## Documentación interactiva

Disponible en `http://localhost:8000/docs` (Swagger UI).

## Desarrollo local (sin Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --reload
```

## Variables de entorno relevantes

| Variable | Default | Descripción |
|---|---|---|
| `QDRANT_COLLECTION` | `tech_manuals` | Nombre de la colección en Qdrant |
| `QDRANT_TOP_K` | `5` | Chunks a recuperar por búsqueda |
| `OLLAMA_MODEL` | `llama3` | Modelo de generación |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Modelo de embeddings |
| `MAX_CONTEXT_CHARS` | `4000` | Límite de caracteres en el contexto |
| `MAX_HISTORY_MESSAGES` | `6` | Turnos de historial a incluir |
| `REDIS_TTL_SECONDS` | `3600` | TTL de sesiones en Redis |
