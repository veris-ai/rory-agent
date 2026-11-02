# Rory on PolyAI's Dialog-RSN-1 realtime API as a Veris Bench image candidate.
#
# The bench platform runs this container directly: the engine injects the World's twin
# base URLs (STRIPE_API_BASE, FINERACT_API_BASE) plus the candidate's env, polls
# GET /health on :8008 for readiness, and opens the voice_ws session at ws://<pod>:8008/voice.
#
#   docker build --platform linux/amd64 -f polyai-dialog-rsn-1/Dockerfile -t rory-polyai-dialog-rsn-1 .
#
# Built from the repo root. `uv sync --frozen --package` installs only this member and
# rory-core out of the shared lockfile, so no other member's files are needed.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /agent

COPY pyproject.toml uv.lock README.md ./
COPY polyai-dialog-rsn-1/pyproject.toml polyai-dialog-rsn-1/README.md ./polyai-dialog-rsn-1/
COPY rory_tools ./rory_tools
COPY polyai-dialog-rsn-1/rory_polyai_dialog_rsn_1 ./polyai-dialog-rsn-1/rory_polyai_dialog_rsn_1
RUN uv sync --frozen --no-dev --no-editable --package rory-polyai-dialog-rsn-1
RUN uv run --no-sync python -c "import rory_polyai_dialog_rsn_1.web"

EXPOSE 8008
CMD ["uv", "run", "--no-sync", "uvicorn", "rory_polyai_dialog_rsn_1.web:app", "--host", "0.0.0.0", "--port", "8008"]
