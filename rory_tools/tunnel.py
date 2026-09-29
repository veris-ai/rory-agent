"""The public URL a hosted platform POSTs tool calls to.

A hosted platform (Vapi, PolyAI) runs the model on its side and executes tools by calling a URL it can
reach from the internet. A bench pod has no public address, so this module
opens a ``cloudflared`` quick tunnel to the candidate's own port at boot —
before ``/health`` answers, so a pod that cannot get a tunnel fails readiness
rather than every call. Each invocation gets a fresh random
``trycloudflare.com`` hostname with no account, so parallel attempts never
share an endpoint.

``PUBLIC_BASE_URL`` short-circuits all of this: when the platform (or a
developer with their own tunnel) already has a public HTTPS address for this
port, it is used as given and no tunnel is spawned.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from loguru import logger

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_READY_TIMEOUT_S = 30.0
_LOG_PATH = Path("/tmp/cloudflared-tunnel.log")


def public_base_url(port: int) -> tuple[str, Optional[subprocess.Popen]]:
    """``(base_url, tunnel_process)``; the process is ``None`` when the URL was given."""
    preset = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if preset:
        if not preset.startswith("https://"):
            raise RuntimeError(f"PUBLIC_BASE_URL must be an absolute https:// URL, got {preset!r}")
        logger.info(f"[tunnel] using PUBLIC_BASE_URL={preset}")
        return preset, None

    logger.info(f"[tunnel] PUBLIC_BASE_URL not set — opening a cloudflared quick tunnel to :{port}")
    log_fh = _LOG_PATH.open("wb")
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(0.5)
        log = _LOG_PATH.read_text(errors="replace")
        if proc.poll() is not None:
            raise RuntimeError(f"cloudflared exited code={proc.returncode}. log tail:\n{log[-1500:]}")
        match = _URL_RE.search(log)
        if match:
            logger.info(f"[tunnel] ready: {match.group(0)}")
            return match.group(0), proc

    proc.terminate()
    raise RuntimeError(
        f"cloudflared did not print a trycloudflare URL within {_READY_TIMEOUT_S:.0f}s. "
        f"log tail:\n{_LOG_PATH.read_text(errors='replace')[-1500:]}"
    )
