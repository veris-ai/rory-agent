#!/usr/bin/env bash
# Launch the three processes that make up the LiveKit candidate:
#   1. livekit-server  — the SFU, on localhost:7880
#   2. the Agents worker (rory_livekit.agent) — connects out to the SFU and is
#      dispatched into every room that gets created
#   3. the FastAPI voice_ws bridge (rory_livekit.web) — serves /voice and
#      /health on :PORT and relays PCM16 audio into and out of a LiveKit room
#
# Ordering matters. A caller opening /voice is what creates a room, and the SFU
# can only dispatch the agent into that room if the worker has already
# registered. So: start the SFU, wait for it, start the worker, wait for it to
# actually register (by tailing its log for "registered worker"), and only then
# bring up the bridge. The bench polls /health for readiness, so the bridge
# coming up last is what makes "healthy" mean "can answer".
#
# Supervision: all three run in the background; a trap + `wait -n` brings the
# container down as a unit when any one dies. Deliberately no `set -e` — it
# would kill the shell on the first child's non-zero exit before cleanup runs.

export PORT="${PORT:-8008}"
export LIVEKIT_URL="${LIVEKIT_URL:-ws://localhost:7880}"
export LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
export LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
# The worker's log is redirected to a file and grepped below; block-buffered
# stdout would hide the registration line until the buffer flushed.
export PYTHONUNBUFFERED=1
# DEBUG for livekit.agents and its plugins (the typer CLI reads this env var).
# A LiveKit call can join, never speak, and run to the actor's 600 s cap with
# nothing at INFO but the greeting announcement; the TTS/LLM request
# lines that would name the hang are DEBUG.
export LIVEKIT_LOG_LEVEL="${LIVEKIT_LOG_LEVEL:-DEBUG}"

wait_for_port() {
  local host="$1" port="$2" tries="${3:-100}"
  for _ in $(seq 1 "$tries"); do
    (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
    sleep 0.2
  done
  return 1
}

# 1. SFU in dev mode (devkey/secret, ws://, no TLS).
livekit-server --dev --bind 0.0.0.0 &
LK_PID=$!

if ! wait_for_port localhost 7880; then
  echo "[start] livekit-server never came up on :7880" >&2
  kill "$LK_PID" 2>/dev/null
  exit 1
fi
echo "[start] livekit-server ready on :7880"

# 2. Agents worker. Its log is captured so the bridge can be gated on actual
# registration, and mirrored to stdout so it still lands in the container log.
# The file is created here, up front: the worker's redirect is opened in the
# child after the fork, so a tail started right behind it can find no file
# yet, exit 1, and hand that status to the supervisor below.
: > /tmp/worker.log
uv run --no-sync python -m rory_livekit.agent start > /tmp/worker.log 2>&1 &
WK_PID=$!
tail -f /tmp/worker.log 2>/dev/null &
TAIL_PID=$!

# Under load the worker can take 10 s+ to register. Give it two minutes, and
# fail the boot rather than open the bridge over a worker that is not there —
# a pod that dies is a candidate failure the bench can see; a bridge with no
# agent behind it is a hundred silent calls.
echo "[start] waiting for the agent worker to register with the SFU..."
registered=0
for _ in $(seq 1 240); do
  grep -q "registered worker" /tmp/worker.log 2>/dev/null && { registered=1; break; }
  kill -0 "$WK_PID" 2>/dev/null || break
  sleep 0.5
done
if [ "$registered" -ne 1 ]; then
  echo "[start] agent worker never registered with the SFU" >&2
  kill "$LK_PID" "$WK_PID" "$TAIL_PID" 2>/dev/null
  exit 1
fi
echo "[start] worker registered — bringing up voice_ws bridge"

# 3. voice_ws bridge.
uv run --no-sync uvicorn rory_livekit.web:app --host 0.0.0.0 --port "$PORT" &
WEB_PID=$!

cleanup() { kill "$LK_PID" "$WK_PID" "$WEB_PID" "$TAIL_PID" 2>/dev/null || true; }
trap 'cleanup; exit 143' TERM INT

# The three peers, by pid: the log mirror is not one of them.
wait -n "$LK_PID" "$WK_PID" "$WEB_PID"
status=$?
echo "[start] a peer exited (status=$status) — shutting down siblings"
cleanup
wait || true
exit "$status"
