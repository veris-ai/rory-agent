# Rory on mistral as a Veris Bench image candidate.
#
# The bench platform runs this container directly: the engine injects the World's twin
# base URLs (STRIPE_API_BASE, FINERACT_API_BASE) plus the candidate's env, polls
# GET /health on :8008 for readiness, and opens the voice_ws session at ws://<pod>:8008/voice.
#
#   docker build --platform linux/amd64 -f mistral/Dockerfile -t rory-mistral .
#
# Built from the repo root. One transport per image: `uv sync --package` installs only
# this member and rory-core out of the shared lockfile. Every member's pyproject is copied
# because uv validates the whole workspace against the lock, but only this transport's
# source ships.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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
COPY grok-cascade/pyproject.toml grok-cascade/README.md ./grok-cascade/
COPY rory_tools ./rory_tools
COPY mistral/rory_mistral ./mistral/rory_mistral
RUN uv sync --frozen --no-dev --no-editable --package rory-mistral
RUN uv run --no-sync python -c "import rory_mistral.web; from rory_tools.prompt import load_agent_prompt; assert load_agent_prompt()"

EXPOSE 8008
CMD ["uv", "run", "--no-sync", "uvicorn", "rory_mistral.web:app", "--host", "0.0.0.0", "--port", "8008"]
