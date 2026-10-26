"""Scoperta automatica dell'ultimo log agente (F2, piano multi-log).

Solo filesystem: mai eccezioni per directory assenti o file svaniti — un tool
non installato significa semplicemente nessuna specifica. Il file più recente
vince per mtime. Nessuna lettura del contenuto: il tail apre solo il path
restituito.
"""

import glob
import os

CLAUDE_DIR = '.claude'
CODEX_DIR = '.codex'


def _cwd_slug(project_cwd: str) -> str:
    """Slug Claude Code di una cwd: '/home/u/proj' → '-home-u-proj'."""
    return project_cwd.replace('/', '-')


def _most_recent(matches: list[str]) -> str | None:
    try:
        return max(matches, key=os.path.getmtime, default=None)
    except OSError:
        return None  # un file è svanito fra glob e stat: nessuna scoperta


def last_claude_session(project_cwd: str | None = None, home: str | None = None) -> str | None:
    """Ultima sessione Claude Code del progetto: ~/.claude/projects/<slug>/*.jsonl."""
    try:
        base = os.path.join(home or os.path.expanduser('~'), CLAUDE_DIR, 'projects')
        cwd = project_cwd if project_cwd is not None else os.getcwd()
        matches = glob.glob(os.path.join(base, _cwd_slug(cwd), '*.jsonl'))
    except OSError:
        return None
    return _most_recent(matches)


def last_codex_session(home: str | None = None) -> str | None:
    """Ultima sessione Codex: ~/.codex/sessions/**/rollout-*.jsonl (ricorsivo)."""
    try:
        base = os.path.join(home or os.path.expanduser('~'), CODEX_DIR, 'sessions')
        matches = glob.glob(os.path.join(base, '**', 'rollout-*.jsonl'), recursive=True)
    except OSError:
        return None
    return _most_recent(matches)


def discover_auto_last(
    tools,
    project_cwd: str | None = None,
    home: str | None = None,
) -> list[tuple[str, str]]:
    """Specifiche (path, formato) per i tool richiesti, nell'ordine richiesto.

    ``tools`` è una sequenza di nomi ('claude', 'codex'); i tool senza sessione
    trovata sono semplicemente assenti dal risultato.
    """
    specs: list[tuple[str, str]] = []
    for tool in tools:
        if tool == 'claude':
            path = last_claude_session(project_cwd, home)
            if path:
                specs.append((path, 'claude-jsonl'))
        elif tool == 'codex':
            path = last_codex_session(home)
            if path:
                specs.append((path, 'codex-jsonl'))
    return specs
