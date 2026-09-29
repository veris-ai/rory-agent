# Rory on PolyAI Agent Studio as a Veris Bench image candidate.
#
# The bench platform runs this container directly: the engine injects the World's twin
# base URLs (STRIPE_API_BASE, FINERACT_API_BASE) plus the candidate's env, polls
# GET /health on :8008 for readiness, and opens the voice_ws session at ws://<pod>:8008/voice.
#
#   docker build --platform linux/amd64 -f polyai/Dockerfile -t rory-polyai .
#
# Built from the repo root. One transport per image: `uv sync --package` installs only
# this member and rory-core out of the shared lockfile. Every member's pyproject is copied
# because uv validates the whole workspace against the lock, but only this transport's
# source ships.
#
# This image also carries two tools the transport shells out to:
#   - `poly`, the PolyAI ADK CLI, which pushes Rory onto the Agent Studio project at boot.
#     The ADK requires Python 3.14 and rory-core stops at 3.14, so it is installed as an
#     isolated uv tool on a managed interpreter rather than into the workspace.
#   - `cloudflared`. Agent Studio functions execute in PolyAI's cloud and POST tool calls
#     to a public HTTPS URL; the bench gives a candidate no public address, so the transport
#     opens a quick tunnel to :8008 at boot. See rory_tools/tunnel.py.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ADD --chmod=755 https://github.com/cloudflare/cloudflared/releases/download/2026.6.0/cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Keep the managed interpreter and the tool outside /root so a non-root
# runtime user can traverse them.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin
RUN uv tool install --python 3.14 polyai-adk==0.57.0 \
    && chmod -R a+rX /opt/uv-python /opt/uv-tools \
    && PYTHONWARNINGS=ignore poly --version

WORKDIR /agent

COPY pyproject.toml uv.lock README.md ./
COPY pipecat/pyproject.toml pipecat/README.md ./pipecat/
COPY gemini-live/pyproject.toml gemini-live/README.md ./gemini-live/
COPY elevenlabs/pyproject.toml elevenlabs/README.md ./elevenlabs/
COPY deepgram/pyproject.toml deepgram/README.md ./deepgram/
COPY openai-realtime/pyproject.toml openai-realtime/README.md ./openai-realtime/
COPY livekit/pyproject.toml livekit/README.md ./livekit/
COPY mistral/pyproject.toml mistral/README.md ./mistral/
COPY fluxions/pyproject.toml fluxions/README.md ./fluxions/
COPY huggingface/pyproject.toml huggingface/README.md ./huggingface/
COPY grok-voice/pyproject.toml grok-voice/README.md ./grok-voice/
COPY vapi/pyproject.toml vapi/README.md ./vapi/
COPY cartesia/pyproject.toml cartesia/README.md ./cartesia/
COPY gradium/pyproject.toml gradium/README.md ./gradium/
COPY polyai/pyproject.toml polyai/README.md ./polyai/
COPY gpt-live/pyproject.toml gpt-live/README.md ./gpt-live/
COPY rory_tools ./rory_tools
COPY polyai/rory_polyai ./polyai/rory_polyai
RUN uv sync --frozen --no-dev --no-editable --package rory-polyai
RUN uv run --no-sync python -c "import rory_polyai.web; from rory_tools.prompt import load_agent_prompt; assert load_agent_prompt()" \
    && cloudflared --version

EXPOSE 8008
CMD ["uv", "run", "--no-sync", "uvicorn", "rory_polyai.web:app", "--host", "0.0.0.0", "--port", "8008"]
