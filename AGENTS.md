# AGENTS.md — Hippocampus Project

## Qué es Hippocampus

Un gestor de memoria de alto rendimiento para LLMs — capa externa entre el usuario y cualquier modelo de lenguaje, que gestiona contexto de forma semántica e inteligente.

Hippocampus no es un modelo. Es infraestructura.

## Arquitectura de roles

```
usuario ←→ LLM grande (Claude, GPT, cualquier API)
                    ↑
              Hippocampus
              (gestor de memoria — corre local, gratis)
                    ↑
              SmolLM2-360M
              (agente interno — embeddings,
               detección de temas, índice, recuperación)
```

- **SmolLM2 nunca habla con el usuario.** Es el motor interno: liviano, CPU-friendly, acceso total a embeddings.
- **El LLM grande nunca gestiona memoria.** Recibe contexto ya curado. No sabe que Hippocampus existe. No consume tokens en gestión.
- **Todo el trabajo de gestión corre local en CPU, gratis.** Solo se llama al LLM grande para responder.

**Por qué SmolLM2:** los modelos cerrados (Claude, GPT) no exponen embeddings vía API. SmolLM2 sí, y son suficientes como proxy semántico — no se usa como modelo de razonamiento.

## Filosofía de diseño

- Hippocampus es un **colaborador**, no un oráculo
- Si no encuentra una memoria con suficiente confianza, pregunta de forma natural — observa primero, pregunta solo cuando genuinamente no puede inferir
- La incertidumbre no es un fallo — es información que dispara una interacción
- Nunca inventar contexto que no existe
- El usuario es parte del sistema de recuperación, no solo un input
- El LLM grande recibe siempre contexto curado, nunca saturado

## Flujo general

```
conversación activa
        ↓
   SmolLM2 extrae embeddings (embed_tokens, sin forward pass)
        ↓
   detección de cambio de tema (con zona de confianza)
        ↓
   chunk semántico → promedio de embeddings → nodo de memoria (.bin)
        ↓
   índice en RAM / contenido en disco
        ↓
   recuperación selectiva cuando el LLM lo necesita
        ↓
   contexto curado → LLM grande → respuesta
```

## Extracción de embeddings

Usar `embed_tokens` directamente — sin correr el transformer completo:

```python
# correcto — liviano, ~200MB RAM
embeddings = model.model.embed_tokens(inputs.input_ids)

# incorrecto — corre 32 capas, 4-6GB RAM
embeddings = model(**inputs, output_hidden_states=True).hidden_states[0]
```

Los vectores crudos capturan qué palabras son, sin contexto posicional. Para detección de tema es suficiente y más limpio — el contexto del transformer distorsiona la señal temática (última capa da similitud ~0.95 para todo).

## Detección de cambio de tema — estado actual

**Configuración vigente:** AND entre dos señales sobre chunks de 3 utterances (ventana no-solapada):

```
cambio de tema SI:
    distancia_coseno(chunk_actual, chunk_anterior) > 0.131
    Y
    hay content words nuevas en el chunk
```

Nota: el threshold 0.131 está en **distancia coseno** (`scipy.spatial.distance.cosine` = 1 − similitud), no en similitud directa.

**Resultado en DialSeg711** (20 diálogos, matching ±1 posición): F1=0.658, P=0.507, R=0.937, FP=72, FN=5.

**Zona de confianza (consulta al usuario):** banda de ±0.05 alrededor del threshold. Dentro de la banda → pregunta natural al usuario en vez de decidir solo. Fuera de la banda → decide automáticamente. Con ±0.05: 22% de transiciones generan pregunta, 58.7% de acierto en las decisiones automáticas. Limitación conocida: ese 58.7% sigue siendo bajo: la banda ayuda pero no resuelve el problema de fondo del detector.

**Decisión de alcance:** se cerró el research loop de afinar el detector contra DialSeg711. Las señales léxicas (cohesión, entidades, marcadores de discurso) tienen techo bajo en ese dataset porque es servicio al cliente transaccional, con alternancia user/agent constante y sin marcadores naturales de transición — no representa el caso de uso real de Hippocampus (conversaciones largas, técnicas, con tangentes).

**Próximo paso de validación:** `conversacion_simulada.json` — conversación sintética en inglés de 50 turnos con boundaries anotados manualmente, que incluye un caso de topic drift real (tangente lateral sobre fermentación de levaduras que interrumpe y luego retoma el tema principal). Sirve para probar si el detector distingue una tangente de un cambio de tema permanente, algo que DialSeg711 no puede testear.

## Estructura del nodo

```
nodo {
    # ÍNDICE (en RAM)
    id:                 uint32
    vector_resumen:     float16[N]      # promedio de embeddings del chunk
    tema:               string
    timestamp:          uint32
    tipo:               enum { principal, lateral, sin_desarrollar }

    # RELACIONES
    puntero_anterior:   id | null
    puntero_siguiente:  id | null
    punteros_laterales: id[]
    origen_lateral:     id | null       # de qué contexto nació este nodo

    # REFERENCIA A DISCO
    archivo:            "memoria_NNN.bin"
}
```

## Formato del .bin de contenido

Opción C — arquitectura final, implementación incremental:

- `embedding_promedio` — float16[N], promedio de embeddings del chunk. Implementado.
- `kv_cache` — placeholder vacío. Pendiente, se implementa cuando el embedding promedio esté validado en producción.

```python
import numpy as np
np.save("memoria_001.bin", embedding_promedio)
embedding = np.load("memoria_001.bin")
```

## Estrategia de búsqueda dual

- **Vectorial** — similitud coseno del query contra vector_resumen de cada nodo
- **Navegación de grafo** — cuando hay contexto, recorrer punteros directamente (más rápido y preciso que búsqueda vectorial cuando aplica)

## Niveles de confianza en recuperación

```
alta   →  carga la memoria directamente
media  →  pregunta natural al usuario para confirmar
baja   →  "no encuentro el contexto, ¿me ayudás a ubicarlo?"
```

## Hardware target

- Ryzen 5 serie 5000, CPU-only, 8GB RAM
- Agente interno: SmolLM2-360M
- Stack: Python + HuggingFace Transformers + torch (CPU)
- Compatible con cualquier LLM grande vía API

## Estado

- [x] Arquitectura conceptual y roles definidos
- [x] Filosofía de diseño
- [x] Estructura del nodo
- [x] Formato del .bin — Opción C, incremental
- [x] Fix de embeddings — embed_tokens sin forward pass
- [x] Detección de cambio de tema — evaluado contra DialSeg711, F1=0.658
- [x] Zona de confianza para consultas al usuario — ±0.05 calibrado
- [x] Conversación simulada en inglés creada (conversacion_simulada.json)
- [ ] Evaluar detector contra conversacion_simulada.json
- [ ] Índice en RAM
- [ ] Navegación de grafo
- [ ] Integración con LLM grande vía API
- [ ] Versión en español

## Limitaciones conocidas

**Granularidad de chunks por utterances, no por tokens.** La ventana fija de 3 utterances funciona en DialSeg711 porque los mensajes son de longitud uniforme. En conversaciones reales con mensajes muy largos o muy cortos puede fallar — un mensaje largo mezcla subtemas en un embedding promedio difuso, mensajes muy cortos no tienen suficiente contenido semántico. Solución propuesta, pendiente de calibrar: acumular utterances hasta llegar a N tokens (rango típico en literatura: 50-150) en vez de un número fijo de mensajes.

**Acierto automático bajo incluso fuera de la zona de confianza (58.7%).** La banda reduce las preguntas innecesarias pero no resuelve que el detector de base siga siendo poco confiable. Pendiente de revisar con datos más representativos del uso real (ver conversacion_simulada.json).

**DialSeg711 no representa el caso de uso real.** Es útil como benchmark estandarizado y comparable con literatura, pero es diálogo transaccional corto. La validación real debe hacerse contra conversaciones largas y técnicas como las que Hippocampus va a procesar en producción.

## Historial de exploración — señales de detección evaluadas

Dataset: DialSeg711, 20 diálogos, ventana no-solapada=3 utterances, matching ±1 posición.

| Configuración | F1 | P | R | FP | FN |
|---|---|---|---|---|---|
| Baseline (embedding sola) | 0.617 | 0.446 | 1.000 | 98 | 0 |
| + Discourse markers | 0.617 | 0.446 | 1.000 | 98 | 0 |
| + Lexical cohesion (AND) | 0.633 | 0.476 | 0.975 | 92 | 2 |
| + Lexical cohesion (boost) | 0.621 | 0.447 | 0.975 | 94 | 2 |
| **+ Content words nuevas (AND)** | **0.658** | **0.507** | **0.937** | **72** | **5** |
| + Entidades nuevas (AND) | 0.627 | 0.459 | 0.987 | 92 | 1 |
| + Promedio CW + Ent (AND) | 0.641 | 0.484 | 0.949 | 80 | 4 |
| + OR interno (AND) | 0.637 | 0.490 | 0.911 | 75 | 7 |

Señales descartadas tras evaluación: discourse markers solos no aportan en este dataset (~3.4% de cobertura). Score combinado multiplicativo (sim × jaccard, en ambas direcciones) probado y descartado — peor que el filtro AND simple. Cohesión léxica sola tiene techo bajo porque comparte stopwords entre chunks de cualquier tema.