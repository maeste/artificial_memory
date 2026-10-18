"""Ranking esatto: coseno, calibrazione per formato e propagazione sinaptica.

Piano ST3 (untracked_doc/02_AGENT_1_STORAGE.md), matematica pura: nessun I/O e
nessuna mutazione dei vettori. Lo score è calcolato su TUTTI i record
dell'archivio (selezionare un sottoinsieme prima della calibrazione violerebbe
il contratto); le soglie di visualizzazione (min_final, filtro UI) non sono
applicate qui. `rank` resta interno al modulo storage.
"""

import numpy as np

from src.contracts import MemoryRecord, ScoredMemory, SearchSettings

# Passo 2 del piano: calibrazione per formato. Il PDF può contenere testo o
# immagini: è una taratura da valutare, non una correzione universale del gap.
_FIXED_MEAN = 0.30
_FIXED_SCALE = 0.15
_STD_FLOOR = 0.05


def normalized_scores(cosines: np.ndarray) -> np.ndarray:
    """Z del gruppo di formato: deviazione campionaria con floor sotto i 3 record."""
    if len(cosines) >= 3:
        return (cosines - cosines.mean()) / max(float(cosines.std(ddof=0)), _STD_FLOOR)
    return (cosines - _FIXED_MEAN) / _FIXED_SCALE


def cosine_scores(records: list[MemoryRecord], query_vec: list[float]) -> np.ndarray:
    """Passo 1: prodotto matrice-vettore fra vettori unitari, clampato a [-1, 1]."""
    if not records:
        return np.zeros(0)
    matrix = np.asarray([record.embedding for record in records], dtype=np.float64)
    query = np.asarray(query_vec, dtype=np.float64)
    return np.clip(matrix @ query, -1.0, 1.0)


def rank_from_cosines(
    records: list[MemoryRecord],
    cosines,
    edges: list[tuple[str, str, float]],
    settings: SearchSettings,
) -> list[ScoredMemory]:
    """Passi 2-6 con coseni già calcolati (il backend sqlite-vec li fornisce)."""
    if len(cosines) != len(records):
        raise ValueError(f"INVALID_REQUEST: {len(cosines)} coseni per {len(records)} record")
    if not records:
        return []
    cosines = np.asarray(cosines, dtype=np.float64)

    z = np.zeros(len(records))
    by_modality: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_modality.setdefault(record.memory.modality, []).append(index)
    for indices in by_modality.values():
        z[indices] = normalized_scores(cosines[indices])

    # Passo 3: base secondo la modalità richiesta.
    base = z if settings.score_mode == "zscore" else cosines

    # Passo 4: semi attivi — soglia diretta sulla base e coseno minimo sul coseno.
    seeds = [
        index
        for index in range(len(records))
        if base[index] > settings.activation_threshold and cosines[index] >= settings.min_cosine
    ]

    # Passo 5: un solo hop, massimo fra i semi collegati, nessuna somma di versi
    # né ricircolo (il boost non entra nelle base).
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for source_id, target_id, weight in edges:
        adjacency.setdefault(source_id, []).append((target_id, weight))
        adjacency.setdefault(target_id, []).append((source_id, weight))
    id_to_index = {record.memory.id: index for index, record in enumerate(records)}
    boosts = np.zeros(len(records))
    for seed in seeds:
        for neighbor_id, weight in adjacency.get(records[seed].memory.id, ()):
            neighbor = id_to_index.get(neighbor_id)
            if neighbor is None:
                continue
            candidate = settings.gamma * base[seed] * weight
            if candidate > boosts[neighbor]:
                boosts[neighbor] = candidate

    # Passo 6: finale, ordinamento final desc, coseno desc, ID crescente.
    finals = base + boosts
    order = sorted(
        range(len(records)),
        key=lambda index: (-finals[index], -cosines[index], records[index].memory.id),
    )
    return [
        ScoredMemory(
            memory=records[index].memory,
            cosine_score=float(cosines[index]),
            z_score=float(z[index]),
            synaptic_boost=float(boosts[index]),
            final_score=float(finals[index]),
            matched_sources=[],
            winning_source=None,
        )
        for index in order
    ]


def rank(
    records: list[MemoryRecord],
    query_vec: list[float],
    edges: list[tuple[str, str, float]],
    settings: SearchSettings,
) -> list[ScoredMemory]:
    """Composizione NumPy: prodotto matrice-vettore poi passi 2-6."""
    return rank_from_cosines(records, cosine_scores(records, query_vec), edges, settings)
