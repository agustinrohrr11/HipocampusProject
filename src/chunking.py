import re
import numpy as np
from scipy.spatial.distance import cosine as coseno

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "although", "since", "until", "about", "between",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "please",
    "thanks", "thank", "yes", "no", "sure", "ok", "okay", "hello", "hi",
    "hey", "well", "so", "also", "now", "get", "got", "need", "like",
    "want", "would", "could", "going", "go", "went", "come", "came",
    "make", "made", "take", "took", "give", "gave", "tell", "told",
    "ask", "asked", "say", "said", "see", "saw", "know", "knew", "think",
    "thought", "look", "looked", "let", "set", "put", "call", "called",
}


def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - coseno(a.astype(np.float64), b.astype(np.float64))


def extraer_content_words(texto: str) -> set[str]:
    words = re.findall(r"[a-z]+", texto.lower())
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}


def ratio_palabras_nuevas(texto_prev: str, texto_curr: str) -> float:
    prev = extraer_content_words(texto_prev)
    curr = extraer_content_words(texto_curr)
    if not curr:
        return 0.0
    return len(curr - prev) / len(curr)


class DetectorCambioTema:
    def __init__(self, th_embed: float = 0.9, th_nuevas: float = 0.73, ventana: int = 3):
        self.th_embed = th_embed
        self.th_nuevas = th_nuevas
        self.ventana = ventana
        self._buffer_embs: list[np.ndarray] = []
        self._buffer_texts: list[str] = []
        self._prev_centroid: np.ndarray | None = None
        self._prev_cw: set[str] = set()
        self._ult_sim = 1.0
        self._ult_ratio = 0.0

    def es_cambio_tema(self, embedding: np.ndarray, texto: str = "") -> bool:
        self._buffer_embs.append(embedding)
        self._buffer_texts.append(texto)

        if len(self._buffer_embs) < self.ventana:
            return False

        centroid = np.mean(self._buffer_embs, axis=0)
        curr_cw = extraer_content_words(" ".join(self._buffer_texts))
        self._buffer_embs.clear()
        self._buffer_texts.clear()

        if self._prev_centroid is None:
            self._prev_centroid = centroid
            self._prev_cw = curr_cw
            return False

        sim = similitud_coseno(self._prev_centroid, centroid)
        ratio = len(curr_cw - self._prev_cw) / len(curr_cw) if curr_cw else 0.0

        self._ult_sim = sim
        self._ult_ratio = ratio

        cambio = sim < self.th_embed and ratio > self.th_nuevas

        self._prev_centroid = centroid
        self._prev_cw = curr_cw

        return cambio

    def ultima_similitud(self) -> float:
        return self._ult_sim

    def ultimo_ratio(self) -> float:
        return self._ult_ratio

    def reiniciar(self):
        self._buffer_embs.clear()
        self._buffer_texts.clear()
        self._prev_centroid = None
        self._prev_cw.clear()
        self._ult_sim = 1.0
        self._ult_ratio = 0.0
