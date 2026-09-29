# Rory on livekit as a Veris Bench image candidate.
#
# The bench platform runs this container directly: the engine injects the World's twin
# base URLs (STRIPE_API_BASE, FINERACT_API_BASE) plus the candidate's env, polls
# GET /health on :8008 for readiness, and opens the voice_ws session at ws://<pod>:8008/voice.
#
#   docker build --platform linux/amd64 -f livekit/Dockerfile -t rory-livekit .
#
# Built from the repo root. One transport per image: `uv sync --package` installs only
# this member and rory-core out of the shared lockfile. Every member's pyproject is copied
# because uv validates the whole workspace against the lock, but only this transport's
# source ships.
#
# Unlike the other transports this image runs three processes (see start.sh): LiveKit
# is WebRTC end to end and cannot terminate voice_ws itself, so a livekit-server SFU
# runs beside the Agents worker and a bridge that joins the room as the caller.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# livekit-server — the in-container SFU the worker and the /voice bridge both reach on
# localhost:7880. Pinned to a fixed release for reproducible builds (the get.livekit.io
# installer resolves "latest" through the unauthenticated GitHub API, which floats
# between builds and rate-limits on CI).
ARG LIVEKIT_SERVER_VERSION=1.12.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates tar \
    && curl -fsSL "https://github.com/livekit/livekit/releases/download/v${LIVEKIT_SERVER_VERSION}/livekit_${LIVEKIT_SERVER_VERSION}_linux_$(dpkg --print-architecture).tar.gz" \
        | tar -xz -C /usr/local/bin livekit-server \
    && rm -rf /var/lib/apt/lists/*

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
COPY livekit/rory_livekit ./livekit/rory_livekit
RUN uv sync --frozen --no-dev --no-editable --package rory-livekit
RUN uv run --no-sync python -c "import rory_livekit.web; from rory_tools.prompt import load_agent_prompt; assert load_agent_prompt()"
COPY livekit/start.sh ./start.sh

# start.sh defaults the SFU's dev credentials (LIVEKIT_URL/API_KEY/API_SECRET) and the
# bridge port. Only the bridge port is published; the SFU is reached over localhost.
EXPOSE 8008
CMD ["bash", "/agent/start.sh"]
