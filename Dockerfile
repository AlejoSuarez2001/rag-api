FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

EXPOSE 8000

# El entrypoint corre `alembic upgrade head` antes de levantar uvicorn.
# Se invoca con `sh` para no depender del bit +x (en dev el bind-mount pisa permisos).
CMD ["sh", "entrypoint.sh"]
