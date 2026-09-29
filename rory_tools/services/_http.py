"""One request against a vendor, shared by the clients that speak plain HTTP.

Stripe has its own SDK; Fineract is plain httpx. This is the piece the httpx
side needs — reach the vendor, turn a 4xx/5xx into a refusal the model can read
aloud, decode a body that may be empty.

The error text carries the vendor's own status and body, truncated. That is
deliberate: Fineract's refusals are the point of using it (a future-dated
repayment, a command against the wrong state), and an agent that is told only
"something went wrong" cannot explain to a caller why their promise to pay next
Tuesday was not recorded as a payment.
"""

from __future__ import annotations

from typing import Any, Type

import httpx


def vendor_request(
    http: httpx.Client,
    error_cls: Type[Exception],
    vendor: str,
    method: str,
    path: str,
    **kwargs: Any,
) -> Any:
    try:
        resp = http.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise error_cls(f"could not reach {vendor}: {exc}") from exc
    if resp.status_code >= 400:
        raise error_cls(f"{vendor} returned {resp.status_code}: {resp.text[:300]}")
    return resp.json() if resp.content else {}
