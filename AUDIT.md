● Buena observación. Hay dos formas de usar book y chapter para mejorar el retrieval:

  Opción A — Boosting (recomendada)

  No filtrar, sino darle más peso a chunks del mismo book/chapter que los resultados más relevantes. Si los top resultados son del libro "Infraestructura", probablemente otros chunks
  de ese libro también sean relevantes.

  Qdrant permite esto con QueryResponse + boost por payload, pero es complejo de implementar.

  Opción B — Filtrado contextual

  Si la pregunta menciona explícitamente un libro o capítulo, filtrar la búsqueda:

   # "¿Cómo reinicio el servidor en Infraestructura?"
   Filter(must=[FieldCondition(key="book", match=MatchValue(value="Infraestructura"))])

  Requiere detección del libro/chapter en la query — complejo.

  Opción C — Enriquecer el texto del chunk antes de embedear

  La más simple y efectiva. En lugar de embedear solo el content, embedear:

   # En ingestion — EmbeddingService
   texto_enriquecido = f"[{chunk.book} › {chunk.chapter}] {chunk.title}\n{chunk.content}"

  Así el vector ya contiene el contexto jerárquico y la búsqueda por similitud lo considera naturalmente.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

  La C es la mejor relación costo/beneficio — un cambio pequeño en ingestion que mejora directamente la calidad del embedding sin tocar el retrieval. ¿La implementamos?
