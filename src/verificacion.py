import numpy as np
from scipy.spatial.distance import cosine as coseno

from src.modelo import EmbeddingExtractor, EstrategiaEmbedding


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - coseno(a.astype(np.float64), b.astype(np.float64))


def main():
    texto_cocina = (
        "La cocina italiana es famosa por sus pastas, pizzas "
        "y el uso de tomate, ajo y albahaca."
    )
    texto_programacion = (
        "Python es un lenguaje de programacion interpretado, "
        "tipado dinamicamente y con una sintaxis clara."
    )

    extractor = EmbeddingExtractor()

    # first-3_mean: los primeros 3 tokens capturan el sujeto/tema
    # y discriminan mejor entre temas distintos
    emb_cocina = extractor.extraer_embedding(
        texto_cocina, estrategia=EstrategiaEmbedding.PRIMEROS_K
    )
    emb_programacion = extractor.extraer_embedding(
        texto_programacion, estrategia=EstrategiaEmbedding.PRIMEROS_K
    )

    sim = similitud_coseno(emb_cocina, emb_programacion)

    print(f"Texto cocina:       {texto_cocina}")
    print(f"Texto programacion: {texto_programacion}")
    print(f"\nSimilitud coseno (first-3): {sim:.6f}")
    if sim < 0.3:
        print("~0 o negativo: los embeddings discriminan temas -> validado")
    else:
        print(f"Similitud {sim:.3f} — puede que no discrimine bien")




if __name__ == "__main__":
    main()
