from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TipoNodo(Enum):
    PRINCIPAL = "principal"
    LATERAL = "lateral"
    SIN_DESARROLLAR = "sin_desarrollar"


@dataclass
class NodoMemoria:
    id: int
    vector_resumen: np.ndarray
    vector_primero: np.ndarray
    tema: str
    timestamp: int
    tipo: TipoNodo

    puntero_anterior: Optional[int] = None
    puntero_siguiente: Optional[int] = None
    punteros_laterales: list[int] = field(default_factory=list)
    origen_lateral: Optional[int] = None

    archivo: str = ""
