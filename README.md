# Hybrid RAG Support API

Backend para responder preguntas sobre manuales de soporte técnico usando Hybrid Retrieval-Augmented Generation.

## Stack

| Componente | Tecnología |
|---|---|
| API | FastAPI |
| Validación | Pydantic v2 |
| Auth | JWT Bearer + JWKS |
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
- NVIDIA Container Toolkit instalado si `rag-api` va a correr con GPU dentro de Docker
- Ollama corriendo con acceso a GPU y con los modelos descargados:
  ```bash
  ollama pull llama3
  ```
- Colección `tech_manuals` ya cargada en Qdrant con índice de texto en el campo `text`
- Configurar autenticación JWT:
  - `AUTH_CERTS`: URL del JWKS
  - `AUTH_SERVER_ISSUER`: issuer esperado del token
  - `KEYCLOAK_CLIENTID`: cliente cuyos roles se validan (default `api-rag`)

## Inicio rápido

```bash
# 0. Crear la red compartida para que rag-api vea a qdrant aunque estén en composes distintos
docker network create rag-shared

# 1. Clonar y configurar
cp .env.example .env
# Qdrant se resuelve por la red Docker compartida como `qdrant`.
# Ollama, si corre fuera de Docker, se consume vía host.docker.internal.
# Completar AUTH_CERTS y AUTH_SERVER_ISSUER para habilitar JWT.

# 2. Levantar servicios con Docker Compose
docker compose up -d

# 3. Verificar que todo esté listo
curl http://localhost:8000/api/v1/health/ready
```

## Ejecucion con GPU

`rag-api` usa GPU para:

- embeddings locales (`EMBEDDING_DEVICE=cpu` por defecto para ahorrar VRAM)
- reranking (`RERANKER_DEVICE=cpu` por defecto para ahorrar VRAM)

El `docker-compose.yml` ya solicita acceso a NVIDIA para el contenedor `api`. Antes de levantarlo, verificá que Docker vea la GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

Ollama no corre dentro de este compose. Si querés generación en GPU, el proceso o contenedor de Ollama también tiene que estar levantado con acceso a NVIDIA y accesible desde `OLLAMA_BASE_URL`.

## Red compartida con Qdrant

Si `rag-api` y `rag_ingestion_service` corren en composes distintos, ambos deben compartir la red externa `rag-shared`.

- `rag-api` se conecta a esa red con el servicio `api`
- `rag_ingestion_service` expone `qdrant` en esa misma red
- desde `rag-api`, el host a usar es `QDRANT_HOST=qdrant`

## Uso de la API

Todas las rutas bajo `/api/v1` requieren un JWT Bearer válido.

- `/api/v1/chat` exige `ROLE_CHATEAR_RAG`
- `/api/v1/health` y `/api/v1/health/ready` exigen `ROLE_CHECK_RAG`

### POST /api/v1/chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer <jwt>" \
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

```bash
curl http://localhost:8000/api/v1/health/ready \
  -H "Authorization: Bearer <jwt>"
```

## Documentación interactiva

Disponible en `http://localhost:8000/docs` (Swagger UI).

## Desarrollo local (sin Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Para desarrollo fuera de Docker, ajustá:
# REDIS_HOST=localhost
# QDRANT_HOST=localhost
# OLLAMA_BASE_URL=http://localhost:11434
uvicorn app.main:app --reload
```

## Variables de entorno relevantes

| Variable | Default | Descripción |
|---|---|---|
| `QDRANT_COLLECTION` | `tech_manuals` | Nombre de la colección en Qdrant |
| `QDRANT_TOP_K` | `5` | Chunks a recuperar por búsqueda |
| `OLLAMA_MODEL` | `llama3` | Modelo de generación |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | URL de Ollama vista desde el contenedor `api` |
| `AUTH_CERTS` | vacío | URL del JWKS para validar JWT |
| `AUTH_SERVER_ISSUER` | vacío | Issuer esperado del JWT |
| `KEYCLOAK_CLIENTID` | `api-rag` | Cliente dentro de `resource_access` desde donde se leen los roles |
| `EMBEDDING_DEVICE` | `cpu` | Device para embeddings locales |
| `RERANKER_DEVICE` | `cpu` | Device para el cross-encoder de reranking |
| `MAX_CONTEXT_CHARS` | `4000` | Límite de caracteres en el contexto |
| `MAX_HISTORY_MESSAGES` | `6` | Turnos de historial a incluir |
| `REDIS_TTL_SECONDS` | `3600` | TTL de sesiones en Redis |
