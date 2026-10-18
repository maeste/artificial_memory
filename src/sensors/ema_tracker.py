"""EMA per sorgente — Sensori (contratti §4/§5).

Una media mobile esponenziale normalizzata per ciascuna sorgente live:
``normalize(alpha*new + (1-alpha)*previous)``. Le sorgenti non si mediano
insieme: ogni tracker è proprietario di un solo flusso.
"""

import math

import numpy as np

from src.contracts import EMBED_DIM


class EMATracker:
    """Stato EMA di una sorgente. Primo vettore copiato, aggiornamenti normalizzati."""

    def __init__(self, alpha: float = 0.35):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError(f'INVALID_REQUEST: alpha non numerico: {alpha!r}')
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError(f'INVALID_REQUEST: alpha fuori da (0, 1]: {alpha!r}')
        self._alpha = float(alpha)
        self._state: np.ndarray | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def has_state(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> list[float] | None:
        """Vettore EMA corrente, o None se nessun embedding è mai riuscito."""
        return None if self._state is None else self._state.tolist()

    def update(self, vector: list[float]) -> list[float]:
        """Valida e assimila un embedding; restituisce il nuovo stato normalizzato.

        Un vettore invalido lascia lo stato invariato. Il contenitore del
        chiamante non viene modificato.
        """
        arr = self._validate(vector)
        if self._state is None:
            merged = arr.copy()
        else:
            merged = self._alpha * arr + (1.0 - self._alpha) * self._state
        norm = float(np.linalg.norm(merged))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError('ZERO_NORM_VECTOR: combinazione EMA a norma nulla')
        self._state = merged / norm
        return self._state.tolist()

    def reset(self) -> None:
        """Torna a stato assente: il prossimo update è un primo vettore."""
        self._state = None

    @staticmethod
    def _validate(vector: list[float]) -> np.ndarray:
        if not isinstance(vector, (list, tuple)):
            raise ValueError(f'INVALID_REQUEST: il vettore non è una lista: {type(vector).__name__}')
        if len(vector) != EMBED_DIM:
            raise ValueError(f'INVALID_VECTOR_DIMENSION: attesi {EMBED_DIM} componenti, ricevute {len(vector)}')
        for component in vector:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise ValueError(f'INVALID_REQUEST: componente non numerica: {component!r}')
            if not math.isfinite(component):
                raise ValueError(f'NON_FINITE_VECTOR: componente non finita: {component!r}')
        return np.asarray(vector, dtype=np.float64)
