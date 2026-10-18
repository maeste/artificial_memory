#!/usr/bin/env bash
# Smoke e2e reale headless (Orchestrator, O3/O5): avvio completo con backend reale,
# snapshot discovery, suggerimento da messaggio agente reale, arresto pulito.
# Uso: bash scripts/smoke_headless.sh [--db PATH]
set -u
cd "$(dirname "$0")/.."

DB="${2:-./data/brain-real.db}"
RUNTIME_ROOT="${XDG_RUNTIME_DIR:-/tmp}/artificial-memory"
LOG="$(mktemp /tmp/am-smoke-agent-XXXX.log)"
: > "$LOG"
REPORT="$(mktemp /tmp/am-smoke-report-XXXX.json)"
echo '{"phase":"start","db":"'"$DB"'"}'

uv run python run_poc.py --watch-dir ./data/watch-corpus --backend real --headless --no-mic \
    --agent-log "$LOG" --agent-format plain --db "$DB" > /tmp/am-smoke-launcher.out 2>&1 &
LAUNCHER=$!

RUN_DIR=""
for _ in $(seq 1 150); do
  CAND="$(ls -td "$RUNTIME_ROOT"/*/ 2>/dev/null | head -1)"
  if [ -n "$CAND" ] && grep -q '"runner"' "$CAND/state.json" 2>/dev/null; then
    RUN_DIR="${CAND%/}"
    break
  fi
  kill -0 $LAUNCHER 2>/dev/null || break
  sleep 2
done
if [ -z "$RUN_DIR" ]; then
  echo '{"phase":"FAIL","error":"runtime non pronto","launcher_log":"'"$(tail -3 /tmp/am-smoke-launcher.out | tr '\n' ' ')"'"}'
  kill $LAUNCHER 2>/dev/null; wait $LAUNCHER 2>/dev/null
  exit 1
fi
echo '{"phase":"runtime_up","run_dir":"'"$RUN_DIR"'","pids":'$(python3 -c "import json;print(json.dumps(json.load(open('$RUN_DIR/state.json'))['pids']))")'}'

# 1. snapshot discovery
uv run python -m src.control --runtime "$RUN_DIR" snapshot > /tmp/am-smoke-snap1.json 2>&1
python3 - <<PY
import json
resp = json.load(open('/tmp/am-smoke-snap1.json'))
state = resp.get('state') or {}
print(json.dumps({'phase': 'snapshot_discovery', 'success': resp.get('success'),
                  'session': state.get('session_id'), 'backend': state.get('backend'),
                  'n_results': len(state.get('results', []))}))
PY

# 2. nuovo messaggio agente reale -> suggerimento atteso (f01.md, caching)
echo "Stiamo ragionando sulla cache: quanto durano i risultati e quando li invalidiamo?" >> "$LOG"
sleep 8
uv run python -m src.control --runtime "$RUN_DIR" snapshot > /tmp/am-smoke-snap2.json 2>&1
python3 - <<PY
import json
resp = json.load(open('/tmp/am-smoke-snap2.json'))
state = resp.get('state') or {}
results = state.get('results', [])
top = results[0] if results else None
print(json.dumps({
    'phase': 'live_suggestion', 'n_results': len(results),
    'top_memory': top['memory']['file_path'].rsplit('/', 1)[-1] if top else None,
    'top_final': top['final_score'] if top else None,
    'top_cosine': top['cosine_score'] if top else None,
    'matched': top['winning_source'] if top else None,
}))
PY

# 3. arresto pulito: SIGINT, nessun orfano, runtime rimosso
PIDS=$(python3 -c "import json;print(' '.join(str(v) for v in json.load(open('$RUN_DIR/state.json'))['pids'].values()))")
kill -INT $LAUNCHER
for _ in $(seq 1 30); do kill -0 $LAUNCHER 2>/dev/null || break; sleep 1; done
kill -0 $LAUNCHER 2>/dev/null && kill -9 $LAUNCHER
wait $LAUNCHER 2>/dev/null
ALIVE=""
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && ALIVE="$ALIVE $pid"; done
echo '{"phase":"teardown","orphans":"'"${ALIVE:-none}"'","runtime_removed":'$([ ! -d "$RUN_DIR" ] && echo true || echo false)',"launcher_exit":'$(grep -o 'exit [0-9]*' /tmp/am-smoke-launcher.out | tail -1 | grep -o '[0-9]*' || echo 0)'}'
rm -f "$LOG" "$REPORT"
