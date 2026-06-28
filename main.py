# Proximo paso: implementar indice en RAM y navegacion de grafo (ver AGENTS.md)

import sys
from pathlib import Path

from src.modelo import EmbeddingExtractor, EstrategiaEmbedding
from src.chunking import DetectorCambioTema


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <archivo> [--th-embed=0.9] [--th-nuevas=0.73]")
        sys.exit(1)

    ruta = sys.argv[1]
    th_embed = 0.9
    th_nuevas = 0.73
    for arg in sys.argv[2:]:
        k, v = arg.split("=")
        if k == "--th-embed":
            th_embed = float(v)
        elif k == "--th-nuevas":
            th_nuevas = float(v)

    with open(ruta, encoding="utf-8") as f:
        texto = f.read()

    extractor = EmbeddingExtractor()
    detector = DetectorCambioTema(th_embed=th_embed, th_nuevas=th_nuevas, ventana=1)

    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    Path("data").mkdir(exist_ok=True)

    for i, parrafo in enumerate(parrafos):
        emb = extractor.extraer_embedding(
            parrafo, estrategia=EstrategiaEmbedding.MEDIA_GLOBAL
        )
        es_cambio = detector.es_cambio_tema(emb, texto=parrafo)
        sim = detector.ultima_similitud()
        ratio = detector.ultimo_ratio()
        label = " [CAMBIO DE TEMA]" if es_cambio else ""
        preview = parrafo[:60].replace("\n", " ")
        print(f"Chunk {i:03d}{label}  (sim={sim:.3f}, ratio_nuevas={ratio:.2f}): {preview}...")

        archivo = f"data/memoria_{i:03d}.bin"
        extractor.guardar(emb, archivo)


if __name__ == "__main__":
    main()
