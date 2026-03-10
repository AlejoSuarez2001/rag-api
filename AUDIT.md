# Audit de consistencia: rag-api + rag_ingestion_service

> Fecha: 2026-03-10

---

## ✅ Lo que está bien

| Aspecto | Detalle |
|---|---|
| Separación de responsabilidades | Limpia. `rag_ingestion_service` solo ingesta, `rag-api` solo consulta. Interfaz via Qdrant. |
| Contratos de datos (field names) | Todos los campos del payload coinciden entre servicios |
| Aliases de compatibilidad | `content → text` y `url → source` correctamente definidos en `db.py` |
| Versiones de dependencias | Sin conflictos entre servicios |
| Documentación | READMEs y `.env.example` completos |

---

## 🔴 Crítico — bloqueante para producción

### Mismatch de dimensiones de embeddings

El problema más grave del sistema. Los dos servicios usan modelos de embedding distintos e incompatibles:

| Servicio | Modelo | Dimensiones | Archivo |
|---|---|---|---|
| `rag_ingestion_service` | `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers) | **768** | `rag_ingestion/config/settings.py` L20 |
| `rag-api` | `nomic-embed-text` (Ollama) | **4096** | `app/config.py` L22 |

**Consecuencia:** Todas las búsquedas vectoriales fallan con error de dimensión. El sistema no funciona en producción.

**Decisión pendiente:** ¿Cuál modelo usamos en ambos?
- Opción A — `sentence-transformers` (768 dims): local, sin dependencia de Ollama, más rápido
- Opción B — `nomic-embed-text` via Ollama (4096 dims): requiere Ollama corriendo en ambos servicios

Una vez decidido, alinear ambos archivos de config y re-ingestar desde cero (la colección de Qdrant queda inválida con vectores de dimensión vieja).

---

## 🟠 Alta prioridad

### 1. Sin validación de dimensiones al iniciar
- El mismatch solo se descubre en la primera query (falla silenciosa)
- **Fix:** Validar dimensión de colección Qdrant al iniciar `retrieval_service`
- **Archivo:** `app/services/retrieval_service.py` L18–24

### 2. Todo error devuelve HTTP 500
- El cliente no puede distinguir errores transitorios de permanentes
- **Fix:** Clasificar errores → 400 (input inválido), 502 (Qdrant/Ollama caído), 504 (timeout)
- **Archivo:** `app/api/routes/chat.py` L28–32

### 3. Sin conteo de tokens en el contexto
- El prompt builder puede superar el límite de tokens del LLM sin saberlo
- **Fix:** Agregar conteo con `tiktoken` antes de enviar al LLM
- **Archivo:** `app/utils/prompt_builder.py` L37–48

### 4. Sin job de re-ingestión programado
- Cambios en BookStack no se sincronizan automáticamente; requiere correr `python main.py ingest` manualmente
- **Fix:** Agregar scheduler (APScheduler o cron) en `rag_ingestion_service/main.py`

### 5. Sin retries al BookStack API
- Falla en errores transitorios de red sin reintentar
- **Fix:** Agregar retry con backoff exponencial
- **Archivo:** `rag_ingestion_service/rag_ingestion/ingest/bookstack.py` L47–51

---

## 🟡 Mejoras menores

| Problema | Archivo | Detalle |
|---|---|---|
| Reranker model hardcodeado | `app/config.py` L36 | Moverlo a variable de entorno |
| Fallback silencioso de Docling | `rag_ingestion/ingest/cleaner.py` L25–27 | Loguear warning cuando se usa el fallback |

---

## Resumen ejecutivo

La arquitectura es **correcta y bien diseñada**. Las responsabilidades están bien separadas y los contratos de datos son consistentes. El único problema real es de **configuración**: los modelos de embedding divergieron entre servicios. Es un fix puntual pero completamente bloqueante.

**Próximo paso obligatorio:** Decidir el modelo de embedding y alinear ambos servicios.
