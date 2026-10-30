# Rory's AgentCore Runtime agent: the reasoning loop on Bedrock, tools through AgentCore Gateway.
#
# AgentCore Runtime runs arm64 containers and calls POST /invocations and GET /ping on :8080.
#
#   docker buildx build --platform linux/arm64 -f agentcore/runtime.Dockerfile -t rory-agentcore-runtime .
#
# Built from the repo root, from the same lockfile and package as the pod image; only the
# entry point differs. agentcore/deploy.py builds and pushes it.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /agent

COPY pyproject.toml uv.lock README.md ./
COPY pipecat/pyproject.toml pipecat/README.md ./pipecat/
COPY gemini-live/pyproject.toml gemini-live/README.md ./gemini-live/
COPY gemini-live-xt-thinking/pyproject.toml gemini-live-xt-thinking/README.md ./gemini-live-xt-thinking/
COPY elevenlabs/pyproject.toml elevenlabs/README.md ./elevenlabs/
COPY elevenlabs-realtime/pyproject.toml elevenlabs-realtime/README.md ./elevenlabs-realtime/
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
COPY smallest/pyproject.toml smallest/README.md ./smallest/
COPY dialt/pyproject.toml dialt/README.md ./dialt/
COPY phonic/pyproject.toml phonic/README.md ./phonic/
COPY agentcore/pyproject.toml agentcore/README.md ./agentcore/
COPY rory_tools ./rory_tools
COPY agentcore/rory_agentcore ./agentcore/rory_agentcore
RUN uv sync --frozen --no-dev --no-editable --package rory-agentcore
RUN AWS_REGION=us-east-1 uv run --no-sync python -c "import rory_agentcore.runtime"

EXPOSE 8080
CMD ["uv", "run", "--no-sync", "uvicorn", "rory_agentcore.runtime:app", "--host", "0.0.0.0", "--port", "8080"]
