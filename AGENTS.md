# AGENTS.md — Hippocampus Project

## Qué es Hippocampus

Un gestor de memoria de alto rendimiento para LLMs — capa externa que se sienta entre el usuario y cualquier modelo de lenguaje, gestionando contexto de forma semántica e inteligente.

Hippocampus no es un modelo. Es infraestructura.

## Arquitectura de roles

```
usuario ←→ LLM grande (Claude, GPT, cualquier API)
                    ↑
              Hippocampus
              (gestor de memoria — corre local, gratis)
                    ↑
              SmolLM2-360M
              (agente interno de Hippocampus —
               procesa embeddings,
               detecta cambios de tema,
               indexa y recupera memorias)
```

**SmolLM2 nunca habla con el usuario.** Es el motor interno de Hippocampus — liviano, rápido, CPU-friendly, acceso total a embeddings. Hace el trabajo sucio de gestión semántica.

**El LLM grande nunca se ensucia las manos.** Recibe contexto ya curado, comprimido y relevante. No sabe que Hippocampus existe. No consume tokens en gestión de memoria.

**Todo el trabajo de gestión corre local en CPU, gratis.** Detección, compresión, indexación y recuperación no cuestan tokens de API. Solo se llama al LLM grande para responder.

## Por qué SmolLM2 como agente interno

Los modelos grandes como Claude no exponen sus embeddings via API. SmolLM2-360M sí — y sus embeddings son suficientemente buenos para gestión semántica de memoria. Se usa como proxy de representación, no como modelo de razonamiento.

## Filosofía de diseño

- Hippocampus es un **colaborador**, no un oráculo
- Si no encuentra una memoria con suficiente confianza, pregunta al usuario de forma natural — observa primero, pregunta solo cuando genuinamente no puede inferir
- La incertidumbre no es un fallo — es información que dispara una interacción
- Nunca inventar contexto que no existe
- El usuario es parte del sistema de recuperación, no solo un input
- El LLM grande recibe siempre contexto curado — nunca contexto saturado o redundante

## Flujo general

```
conversación activa
        ↓
   SmolLM2 extrae embeddings (embed_tokens, sin forward pass)
        ↓
   detección de cambio de tema
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

Los vectores crudos de `embed_tokens` capturan qué palabras son, sin contexto posicional. Para detección de tema es suficiente y más limpio — el contexto del transformer puede distorsionar la señal temática.

## Detección de cambio de tema

Basada en datos reales con ground truth conocido — no en umbrales calibrados a mano.

Dataset de referencia: **DialSeg711** — 711 diálogos en inglés con cambios de tema anotados manualmente.

```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/dialogstudio", "DialSeg711")
```

Señales a evaluar:
- `sim_first3 × sim_media` — combinación actual, validada empíricamente
- Marcadores de discurso explícitos — "by the way", "anyway", "moving on"
- Aparición de sustantivos nuevos no presentes en chunk anterior
- Cohesión léxica — caída en vocabulario compartido entre chunks

Metodología:
1. Correr el detector sobre DialSeg711
2. Comparar detecciones vs anotaciones reales
3. Medir aciertos y fallos
4. Iterar sobre señales hasta mejorar la métrica

Para evitar falsos positivos por topic drift:
- Ventana de confirmación — N chunks consecutivos con similitud baja antes de declarar cambio
- Vector de tema dominante — promedio ponderado de últimos N embeddings

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

- `embedding_promedio` — float16[N], promedio de embeddings del chunk. Se implementa primero.
- `kv_cache` — placeholder vacío. Se implementa cuando el embedding promedio funcione.

```python
import numpy as np
np.save("memoria_001.bin", embedding_promedio)
embedding = np.load("memoria_001.bin")
```

## Estrategia de búsqueda dual

- **Vectorial** — similitud coseno del query contra vector_resumen de cada nodo
- **Navegación de grafo** — cuando hay contexto, recorrer punteros directamente

## Niveles de confianza

```
alta   →  carga la memoria directamente
media  →  pregunta natural al usuario para confirmar
baja   →  "no encuentro el contexto, ¿me ayudás a ubicarlo?"
```

## Hardware target

- Ryzen 5 serie 5000, CPU-only, 8GB RAM
- Agente interno: SmolLM2-360M
- Stack: Python + HuggingFace Transformers + torch (CPU)
- Compatible con cualquier LLM grande via API

## Estado

- [x] Arquitectura conceptual y roles definidos
- [x] Estructura del nodo
- [x] Detección de cambio de tema — metodología definida
- [x] Búsqueda dual
- [x] Filosofía de diseño
- [x] Formato del .bin — Opción C, incremental
- [x] Fix de embeddings — embed_tokens sin forward pass
- [ ] Evaluación contra DialSeg711
- [ ] Índice en RAM
- [ ] Navegación de grafo
- [ ] Integración con LLM grande via API

## Pendientes conceptuales

- Umbral de similitud coseno para detección de cambio de tema
- N para ventana de confirmación de topic drift

## Limitaciones conocidas y pendientes de investigación

### Granularidad de chunks — mensajes de longitud variable

La ventana fija de 3 utterances funciona bien en DialSeg711 porque los mensajes tienen longitud relativamente uniforme. En conversaciones reales con mensajes de longitud muy variable puede fallar:

- Mensaje largo (200 palabras) → embedding promedio difuso, mezcla subtemas
- Mensajes muy cortos ("ok", "sí", "entiendo") → poco contenido semántico para promediar

**Solución propuesta:** chunk por tokens, no por utterances.

```
chunk = acumular utterances hasta llegar a N tokens
```

Así un mensaje largo forma su propio chunk y varios mensajes cortos se agrupan hasta tener suficiente contenido semántico. Rango típico en literatura: 50-150 tokens por chunk. Pendiente de calibrar contra DialSeg711.

## Resultados de evaluación — detección de cambio de tema

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

**Configuración ganadora:** Content words nuevas (AND) — F1=0.658, primera vez Precision >0.5.

**Decisión:** integrar content words nuevas al pipeline y cerrar el research loop de detección. El detector es funcional para avanzar a la siguiente etapa. Los FP residuales se manejarán con la consulta colaborativa al usuario cuando la confianza sea baja.

**Límite alcanzado:** señales léxicas no pueden resolver completamente el ruido por alternancia usuario/agente. Mejoras futuras requieren señales más sofisticadas o más datos de entrenamiento.