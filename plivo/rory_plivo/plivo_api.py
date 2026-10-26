"""Per-pod Plivo provisioning: one Application and one Endpoint, created at boot.

A call placed by an endpoint runs the answer_url of the endpoint's own
Application, so a pod that owns both gets every one of its calls routed back to
itself — no phone number, and no shared answer_url for parallel pods to fight
over. Both are deleted at shutdown.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from dataclasses import dataclass

import httpx
from loguru import logger

# A cloudflared quick tunnel's hostname takes a few seconds to resolve after
# the tunnel reports ready. Plivo must not look it up before then: its resolver
# caches the miss, and the call's answer_url fetch later fails with "Error
# Reaching Answer URL" (the call is answered 486 Busy Here). So provisioning
# first waits until public resolvers answer for the hostname.
PUBLIC_RESOLVERS = ("https://cloudflare-dns.com/dns-query", "https://dns.google/resolve")
DNS_WAIT_S = 120.0
# Plivo validates answer_url when the Application is created.
APP_CREATE_ATTEMPTS = 20
APP_CREATE_RETRY_S = 3.0
# A fresh endpoint's SIP credentials are not accepted for a few seconds.
ENDPOINT_SETTLE_S = 5.0


@dataclass
class Provisioned:
    app_id: str
    endpoint_id: str
    username: str
    password: str


def _client() -> httpx.AsyncClient:
    auth_id = os.environ["PLIVO_AUTH_ID"]
    return httpx.AsyncClient(
        base_url=f"https://api.plivo.com/v1/Account/{auth_id}",
        auth=(auth_id, os.environ["PLIVO_AUTH_TOKEN"]),
        timeout=20,
    )


async def wait_for_public_dns(host: str) -> None:
    """Block until every public resolver returns an A record for ``host``."""
    deadline = asyncio.get_running_loop().time() + DNS_WAIT_S
    async with httpx.AsyncClient(timeout=10, headers={"accept": "application/dns-json"}) as dns:
        while True:
            answers = [
                (await dns.get(resolver, params={"name": host, "type": "A"})).json().get("Answer")
                for resolver in PUBLIC_RESOLVERS
            ]
            if all(answers):
                logger.info(f"[plivo] {host} resolves publicly")
                return
            if asyncio.get_running_loop().time() > deadline:
                raise RuntimeError(f"{host} still not resolvable by {PUBLIC_RESOLVERS} after {DNS_WAIT_S:.0f}s")
            await asyncio.sleep(2)


async def provision(answer_url: str) -> Provisioned:
    await wait_for_public_dns(httpx.URL(answer_url).host)
    tag = uuid.uuid4().hex[:10]
    async with _client() as api:
        for attempt in range(1, APP_CREATE_ATTEMPTS + 1):
            r = await api.post(
                "/Application/",
                json={"app_name": f"rory-{tag}", "answer_url": answer_url, "answer_method": "POST"},
            )
            if r.status_code == 201:
                break
            logger.info(f"[plivo] Application create attempt {attempt} -> {r.status_code} {r.text[:200]}")
            await asyncio.sleep(APP_CREATE_RETRY_S)
        r.raise_for_status()
        app_id = r.json()["app_id"]

        # Plivo's endpoint password rules want upper, lower, digit and symbol.
        password = secrets.token_urlsafe(16) + "Aa1!"
        r = await api.post(
            "/Endpoint/",
            json={"username": "rory", "password": password, "alias": f"rory-{tag}", "app_id": app_id},
        )
        r.raise_for_status()
        endpoint = r.json()

    await asyncio.sleep(ENDPOINT_SETTLE_S)
    logger.info(f"[plivo] provisioned app={app_id} endpoint={endpoint['username']} answer_url={answer_url}")
    return Provisioned(app_id, endpoint["endpoint_id"], endpoint["username"], password)


async def teardown(p: Provisioned) -> None:
    async with _client() as api:
        (await api.delete(f"/Endpoint/{p.endpoint_id}/")).raise_for_status()
        (await api.delete(f"/Application/{p.app_id}/")).raise_for_status()
    logger.info(f"[plivo] deleted app={p.app_id} endpoint={p.username}")
