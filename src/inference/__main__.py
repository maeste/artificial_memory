"""CLI embed/probe dell'inferenza (Task I4).

embed: un embedding da CLI; JSON con SpaceSpec, richiesta, vettore e latenza.
probe: corpus e query dal manifest 08 senza Storage, retrieval coseno NumPy,
bank MemoryRecord per l'import Storage, report con tempi (load/warmup/p50/p95)
e hardware. Il gate preliminare usa soltanto le query di calibrazione C1-C4;
T1-T8 restano riservate al gate finale.

Config JSON (prodotta dall'Orchestratore):
  {"schema_version": 1,
   "space": {"backend": "real|mock", "model_id": ..., "revision": ...,
             "dimension": 256, "preprocessing": "eg2-poc-v1"},
   "device": "auto|cpu|cuda"}
"""

import argparse
import json
import math
import os
import platform
import signal
import sys
import threading
import time
import uuid
from dataclasses import asdict

import numpy as np

from src.contracts import EmbedRequest, MemoryRecord, MemoryView, SpaceSpec
from src.inference.model import EmbeddingGemmaBackend, SemanticHashMock
from src.inference.server import EmbedServer

CONFIG_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
BANK_SCHEMA_VERSION = 1
TOP_K = 5

_FORMAT_BY_EXT = {
    ".md": "markdown",
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".mp4": "video",
}


def load_config(path: str) -> tuple[SpaceSpec, str]:
    """Carica e valida la config probe; errori sono SystemExit con messaggio chiaro."""
    try:
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
    except OSError as exc:
        raise SystemExit(f"config illeggibile: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"config non è JSON valido: {path}: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"config: schema_version atteso {CONFIG_SCHEMA_VERSION}: {path}")
    # campi extra dell'Orchestratore ammessi: space_id è verificato sotto,
    # embed_timeout_ms/startup_timeout_sec appartengono al launcher/server
    unknown = set(config) - {
        "schema_version",
        "space",
        "device",
        "space_id",
        "embed_timeout_ms",
        "startup_timeout_sec",
    }
    if unknown:
        raise SystemExit(f"config: campi sconosciuti {sorted(unknown)}")
    space_raw = config.get("space")
    known_space = {"backend", "model_id", "revision", "dimension", "preprocessing"}
    if not isinstance(space_raw, dict) or set(space_raw) - known_space:
        raise SystemExit("config: spazio mancante o con campi sconosciuti")
    if not {"backend", "model_id", "revision"} <= set(space_raw):
        raise SystemExit("config: spazio richiede backend, model_id, revision")
    space = SpaceSpec(**space_raw)
    expected_space_id = config.get("space_id")
    if expected_space_id is not None and expected_space_id != space.space_id:
        raise SystemExit(
            f"config: space_id non corrisponde allo SpaceSpec: atteso {space.space_id}, trovato {expected_space_id}"
        )
    return space, str(config.get("device", "auto"))


def build_backend(space: SpaceSpec, device: str):
    """Backend dallo SpaceSpec; nessun fallback da reale a mock."""
    if space.backend == "mock":
        return SemanticHashMock(space)
    if space.backend != "real":
        raise SystemExit(f"config: backend non ammesso: {space.backend!r}")
    return EmbeddingGemmaBackend(space, device)


def percentile_nearest_rank(values: list[float], pct: float) -> float:
    """Metodo nearest-rank (08 §3): p50/p95 sui tempi grezzi."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def hardware_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        info["torch"] = "non disponibile"
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    info["mem_total_kb"] = int(line.split()[1])
                    break
    except OSError:
        pass
    return info


def cosine_topk(query: list[float], matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Vettori unitari: il coseno è il prodotto scalare. Nessuna soglia qui."""
    scores = matrix @ np.asarray(query, dtype=np.float32)
    order = np.argsort(scores)[::-1][:k]
    return [(int(index), float(scores[index])) for index in order]


def _load_document_requests(manifest: dict, manifest_dir: str) -> list[tuple[str, MemoryView, EmbedRequest]]:
    relative = manifest.get("document_requests_path", "document_requests.jsonl")
    path = os.path.join(manifest_dir, relative)
    entries: list[tuple[str, MemoryView, EmbedRequest]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            entries.append((entry["file_id"], MemoryView(**entry["memory"]), EmbedRequest(**entry["request"])))
    return entries


def _aggregate(times_sec: list[float]) -> dict:
    if not times_sec:
        return {"n": 0, "raw_sec": [], "p50_ms": None, "p95_ms": None, "total_sec": 0.0}
    return {
        "n": len(times_sec),
        "raw_sec": [round(value, 4) for value in times_sec],
        "p50_ms": round(percentile_nearest_rank(times_sec, 50) * 1000, 2),
        "p95_ms": round(percentile_nearest_rank(times_sec, 95) * 1000, 2),
        "total_sec": round(sum(times_sec), 3),
    }


def _hit_metrics(ranked_hits: list[bool]) -> dict:
    """hit@5 e MRR@5 secondo 08 §3 su una lista di esiti booleani in ordine di rango."""
    ranks = [index + 1 for index, hit in enumerate(ranked_hits[:TOP_K]) if hit]
    return {
        "hit_at_5": (1.0 if ranks else 0.0),
        "mrr_at_5": (1.0 / ranks[0] if ranks else 0.0),
    }


def _file_format(files_by_id: dict, file_id: str) -> str | None:
    entry = files_by_id.get(file_id)
    if entry is None:
        return None
    return _FORMAT_BY_EXT.get(os.path.splitext(entry["path"])[1].lower())


def run_probe(
    config_path: str,
    manifest_path: str,
    report_path: str,
    bank_path: str | None = None,
    split: str | None = None,
    repeats: int = 1,
) -> dict:
    """Esegue la prova: corpus, query, ranking coseno, report e bank opzionale."""
    space, device = load_config(config_path)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    files_by_id = {entry["id"]: entry for entry in manifest["files"]}
    documents = _load_document_requests(manifest, manifest_dir)

    load_start = time.perf_counter()
    backend = build_backend(space, device)
    load_sec = time.perf_counter() - load_start
    warmup_start = time.perf_counter()
    backend.embed(EmbedRequest(request_id="warmup", is_query=True, text_payload="warmup"))
    warmup_sec = time.perf_counter() - warmup_start

    # corpus: le richieste documentali del manifest, già conformi al contratto
    records: list[MemoryRecord] = []
    record_file_ids: list[str] = []
    corpus_times: list[float] = []
    corpus_errors: list[dict] = []
    for file_id, memory, request in documents:
        try:
            start = time.perf_counter()
            vector = backend.embed(request)
            corpus_times.append(time.perf_counter() - start)
        except Exception as exc:  # gli errori contano come fallimenti (08 §4)
            corpus_errors.append({"chunk_id": memory.id, "file_id": file_id, "error": str(exc)})
            continue
        records.append(MemoryRecord(memory=memory, embedding=vector))
        record_file_ids.append(file_id)
    matrix = np.asarray([record.embedding for record in records], dtype=np.float32)

    queries = [q for q in manifest["queries"] if split is None or q["split"] == split]
    query_entries: list[dict] = []
    text_times: list[float] = []
    audio_times: list[float] = []
    for query in queries:
        gold_ids = [item["file_id"] for item in query["relevant"]]
        channels = ["text"]
        if query.get("audio_path"):
            channels.append("audio")
        for channel in channels:
            times: list[float] = []
            vector: list[float] | None = None
            error: str | None = None
            for _ in range(max(1, repeats)):
                if channel == "text":
                    request = EmbedRequest(
                        request_id=f"query:{query['id']}:text", is_query=True, text_payload=query["text"]
                    )
                else:
                    audio_path = os.path.join(manifest_dir, query["audio_path"])
                    request = EmbedRequest(
                        request_id=f"query:{query['id']}:audio", is_query=True, audio_path=audio_path
                    )
                try:
                    start = time.perf_counter()
                    vector = backend.embed(request)
                    times.append(time.perf_counter() - start)
                except Exception as exc:
                    error = str(exc)
                    break
            entry: dict = {
                "query_id": query["id"],
                "split": query["split"],
                "channel": channel,
                "gold_file_ids": gold_ids,
                "times_sec": [round(value, 4) for value in times],
                "error": error,
            }
            if vector is not None:
                ranking = cosine_topk(vector, matrix, TOP_K)
                ranked_ids = [record_file_ids[index] for index, _ in ranking]
                entry["top5"] = [
                    {"rank": position + 1, "file_id": ranked_ids[position], "score": round(score, 4)}
                    for position, (_, score) in enumerate(ranking)
                ]
                entry["hit_at_5"] = any(fid in gold_ids for fid in ranked_ids)
                entry["first_hit_rank"] = next(
                    (position + 1 for position, fid in enumerate(ranked_ids) if fid in gold_ids), None
                )
            query_entries.append(entry)
    for entry in query_entries:
        if entry["channel"] == "text" and entry["times_sec"]:
            text_times.extend(entry["times_sec"])
        if entry["channel"] == "audio" and entry["times_sec"]:
            audio_times.extend(entry["times_sec"])

    metrics = _compute_metrics(query_entries, files_by_id)
    gate = _gate_verdict(space, metrics, corpus_errors)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp": time.time(),
        "space": asdict(space),
        "backend_info": dict(backend.info),
        "hardware": hardware_info(),
        "corpus_checksum": manifest.get("corpus_checksum"),
        "manifest": os.path.abspath(manifest_path),
        "split": split or "all",
        "distractors": "corpus con nomi neutri f01..f16 (26 chunk); nessun nome scelto dai risultati",
        "counts": {"chunks": len(records), "chunk_errors": len(corpus_errors), "queries": len(queries)},
        "corpus_errors": corpus_errors,
        "times": {
            "load_sec": round(load_sec, 3),
            "warmup_embed_sec": round(warmup_sec, 4),
            "corpus_embed": _aggregate(corpus_times),
            "query_text": _aggregate(text_times),
            "query_audio": _aggregate(audio_times),
        },
        "metrics": metrics,
        "gate_fase1_preliminare": gate,
        "queries": query_entries,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    if bank_path is not None:
        bank = {
            "schema_version": BANK_SCHEMA_VERSION,
            "space": asdict(space),
            "records": [{"memory": asdict(record.memory), "embedding": record.embedding} for record in records],
        }
        with open(bank_path, "w", encoding="utf-8") as fh:
            json.dump(bank, fh, ensure_ascii=False)
    return report


def _compute_metrics(query_entries: list[dict], files_by_id: dict) -> dict:
    """Metriche 08 §3 separate per sorgente (testo/audio) e per formato del gold."""
    metrics: dict = {}
    for channel in ("text", "audio"):
        entries = [e for e in query_entries if e["channel"] == channel and e.get("gold_file_ids")]
        hit_values = [bool(e.get("hit_at_5")) for e in entries]
        mrr = sum((1.0 / e["first_hit_rank"]) if e.get("first_hit_rank") else 0.0 for e in entries)
        metrics[channel] = {
            "n": len(entries),
            "hit_at_5": (sum(hit_values) / len(entries)) if entries else None,
            "mrr_at_5": (mrr / len(entries)) if entries else None,
        }
        per_format: dict = {}
        for entry in entries:
            for gold in entry["gold_file_ids"]:
                fmt = _file_format(files_by_id, gold)
                if fmt is None:
                    continue
                bucket = per_format.setdefault(fmt, {"n": 0, "hits": 0})
                bucket["n"] += 1
                if entry.get("hit_at_5"):
                    bucket["hits"] += 1
        metrics[channel]["per_format"] = per_format
    negatives = [e for e in query_entries if not e.get("gold_file_ids")]
    if negatives:
        metrics["negative_diagnostica"] = {
            "n": len({e["query_id"] for e in negatives}),
            "con_card_in_top5": len({e["query_id"] for e in negatives if e.get("hit_at_5")}),
            "nota": "senza soglie: diagnostica, non astensione (gate privo di soglie UI)",
        }
    return metrics


def _gate_verdict(space: SpaceSpec, metrics: dict, corpus_errors: list[dict]) -> dict:
    """Gate preliminare Fase 1: C1-C4 testo E audio 4/4, un hit per formato per sorgente."""
    if space.backend != "real":
        return {"applicabile": False, "motivo": f"backend {space.backend}: gate solo con pesi reali"}
    gate = {"applicabile": True, "corpus_errors": len(corpus_errors) == 0}
    for channel in ("text", "audio"):
        channel_metrics = metrics.get(channel, {})
        hits = channel_metrics.get("hit_at_5")
        formats = channel_metrics.get("per_format", {})
        gate[f"{channel}_4_su_4"] = hits == 1.0
        gate[f"{channel}_formati"] = {fmt: bucket["hits"] > 0 for fmt, bucket in sorted(formats.items())}
        gate[f"{channel}_un_hit_per_formato"] = bool(formats) and all(
            bucket["hits"] > 0 for bucket in formats.values()
        )
    gate["pass"] = (
        gate.get("text_4_su_4", False)
        and gate.get("audio_4_su_4", False)
        and gate.get("text_un_hit_per_formato", False)
        and gate.get("audio_un_hit_per_formato", False)
        and gate.get("corpus_errors", False)
    )
    return gate


def cmd_embed(args: argparse.Namespace) -> None:
    space, device = load_config(args.config)
    backend = build_backend(space, device)
    if args.text is not None:
        request = EmbedRequest(
            request_id=uuid.uuid4().hex,
            is_query=args.query,
            text_payload=args.text,
            context_header=args.context_header,
        )
    elif args.audio is not None:
        request = EmbedRequest(request_id=uuid.uuid4().hex, is_query=True, audio_path=os.path.abspath(args.audio))
    else:
        raise SystemExit("embed: serve --text oppure --audio")
    start = time.perf_counter()
    vector = backend.embed(request)
    latency_ms = (time.perf_counter() - start) * 1000
    output = {
        "schema_version": 1,
        "space": asdict(space),
        "backend_info": dict(backend.info),
        "request": asdict(request),
        "embedding": vector,
        "latency_ms": round(latency_ms, 3),
    }
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    print()


def cmd_probe(args: argparse.Namespace) -> None:
    report = run_probe(
        args.config,
        args.manifest,
        args.report,
        bank_path=args.bank,
        split=args.split,
        repeats=args.repeats,
    )
    gate = report["gate_fase1_preliminare"]
    summary = {
        "report": os.path.abspath(args.report),
        "backend": report["space"]["backend"],
        "chunks": report["counts"]["chunks"],
        "queries": report["counts"]["queries"],
        "hit_at_5_text": report["metrics"]["text"]["hit_at_5"],
        "hit_at_5_audio": report["metrics"]["audio"]["hit_at_5"],
        "p95_query_text_ms": report["times"]["query_text"]["p95_ms"],
        "p95_query_audio_ms": report["times"]["query_audio"]["p95_ms"],
        "gate_fase1_preliminare": gate,
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    print()
    if gate.get("applicabile") and not gate.get("pass"):
        raise SystemExit(2)


# ------------------------------------------------------------------ Task serve


def _log(message: str) -> None:
    print(f"[inference] {time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


def load_serve_config(path: str) -> tuple[SpaceSpec, str, str]:
    """Config di bootstrap del launcher (07 §6: passato INTEGRALMENTE ai figli).

    Il file è condiviso fra server, watcher e runner: le chiavi degli altri
    figli (search, alpha, max_pins, session_*, agent_log, ...) sono accettate e
    ignorate. La strettezza riguarda i campi CHE CONSUMO: space malformato,
    paths.embed_endpoint mancante o space_id non corrispondente → errore.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
    except OSError as exc:
        raise SystemExit(f"config illeggibile: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"config non è JSON valido: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"config non è un oggetto: {path}")
    space_raw = config.get("space")
    known_space = {"backend", "model_id", "revision", "dimension", "preprocessing"}
    if not isinstance(space_raw, dict) or set(space_raw) - known_space:
        raise SystemExit("config: spazio mancante o con campi sconosciuti")
    if not {"backend", "model_id", "revision"} <= set(space_raw):
        raise SystemExit("config: spazio richiede backend, model_id, revision")
    space = SpaceSpec(**space_raw)
    expected_space_id = config.get("space_id")
    if expected_space_id is not None and expected_space_id != space.space_id:
        raise SystemExit(
            f"config: space_id non corrisponde allo SpaceSpec: atteso {space.space_id}, trovato {expected_space_id}"
        )
    paths_raw = config.get("paths")
    if not isinstance(paths_raw, dict):
        raise SystemExit("config: paths mancante (serve paths.embed_endpoint)")
    endpoint = paths_raw.get("embed_endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise SystemExit("config: paths.embed_endpoint mancante o vuoto")
    return space, endpoint, str(config.get("device", "auto"))


def run_serve(config_path: str, poll_interval: float = 0.5) -> int:
    """Entrypoint di servizio: EmbedServer vive fino a SIGTERM/SIGINT.

    Ritorna 0 su arresto pulito, 1 se il caricamento del modello fallisce
    (nessun fallback al mock). Da chiamare nel main thread per i segnali;
    fuori dal main thread i segnali vengono semplicemente ignorati.
    """
    space, endpoint, device = load_serve_config(config_path)
    _log(
        f"avvio servizio: endpoint={endpoint} backend={space.backend} "
        f"model={space.model_id} revision={space.revision[:12]} device={device}"
    )
    if space.backend == "mock":
        mock_backend = SemanticHashMock(space)

        def factory():
            return mock_backend

    else:

        def factory():
            return EmbeddingGemmaBackend(space, device)

    server = EmbedServer(
        endpoint,
        space,
        factory,
        on_ready=lambda: _log("modello pronto (caricato e warmup riuscito)"),
    )
    stop = threading.Event()

    def _handle_signal(signum, _frame):
        _log(f"segnale {signum} ricevuto: arresto")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_signal)
        except ValueError:  # fuori dal main thread (test in-process)
            pass
    server.start()
    _log("REP in ascolto, caricamento modello e warmup in corso")
    try:
        while not stop.is_set():
            if server.load_error is not None:
                _log(f"ERRORE caricamento modello: {server.load_error}")
                return 1
            stop.wait(poll_interval)
    finally:
        server.stop()
    _log("arresto completato")
    return 0


def cmd_serve(args: argparse.Namespace) -> None:
    raise SystemExit(run_serve(args.config))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.inference", description="Inferenza multimodale: embed e probe"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    embed_parser = subparsers.add_parser("embed", help="un embedding: --text oppure --audio")
    embed_parser.add_argument("--config", required=True, help="config JSON con SpaceSpec e device")
    embed_parser.add_argument("--text", help="payload testuale")
    embed_parser.add_argument("--audio", help="path WAV mono 16 kHz")
    embed_parser.add_argument("--query", action="store_true", help="marca la richiesta come query")
    embed_parser.add_argument("--context-header", default="", help="header logico del documento")
    embed_parser.set_defaults(func=cmd_embed)
    probe_parser = subparsers.add_parser("probe", help="corpus + query dal manifest 08, senza Storage")
    probe_parser.add_argument("--config", required=True)
    probe_parser.add_argument("--manifest", required=True)
    probe_parser.add_argument("--report", required=True)
    probe_parser.add_argument("--bank", help="bank vettoriale MemoryRecord per l'import Storage")
    probe_parser.add_argument("--split", choices=["calibration", "test"])
    probe_parser.add_argument("--repeats", type=int, default=1, help="ripetizioni per query (tempi a caldo)")
    probe_parser.set_defaults(func=cmd_probe)
    serve_parser = subparsers.add_parser(
        "serve", help="servizio REP: avvia EmbedServer dal config di bootstrap del launcher"
    )
    serve_parser.add_argument("--config", required=True, help="config JSON del launcher (run_dir/config.json)")
    serve_parser.set_defaults(func=cmd_serve)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
