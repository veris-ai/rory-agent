"""The smallest SIP user agent that can place one call as a Plivo endpoint.

Plivo's browser SDK is JsSIP speaking SIP over a secure WebSocket to
``wss://client.plivo.com/signalling``, logged in as ``<endpoint>@phone.plivo.com``.
This module speaks the same messages from Python: REGISTER, INVITE, ACK and BYE,
each answering one digest challenge. Media is not here; the SDP it carries is
aiortc's (see ``rory_plivo.caller``).

Two details are load-bearing, both found against Plivo's live proxy:

- The ACK for the INVITE's 200 OK goes to the answer's Contact URI with the
  Record-Route set as a Route header (reversed, RFC 3261 §12.1.2). Sent to the
  original Request-URI, the proxy drops it and Plivo retransmits the 200 OK.
- ``Supported: ice,replaces,outbound`` and ``;ob`` on the Contact mirror the SDK.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, field

DOMAIN = "phone.plivo.com"
SIGNALLING_URL = "wss://client.plivo.com/signalling"


def digest_authorization(method: str, uri: str, user: str, password: str, challenge: str) -> str:
    """RFC 2617 digest response for one ``WWW-``/``Proxy-Authenticate`` challenge."""
    realm = re.search(r'realm="([^"]+)"', challenge).group(1)
    nonce = re.search(r'nonce="([^"]+)"', challenge).group(1)
    qop = re.search(r'qop="?([^",]+)', challenge)
    opaque = re.search(r'opaque="([^"]+)"', challenge)
    ha1 = hashlib.md5(f"{user}:{realm}:{password}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    parts = [f'username="{user}"', f'realm="{realm}"', f'nonce="{nonce}"', f'uri="{uri}"', "algorithm=MD5"]
    if qop:
        cnonce, nc = secrets.token_hex(8), "00000001"
        response = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}".encode()).hexdigest()
        parts += ["qop=auth", f'cnonce="{cnonce}"', f"nc={nc}"]
    else:
        response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    parts.append(f'response="{response}"')
    if opaque:
        parts.append(f'opaque="{opaque.group(1)}"')
    return "Digest " + ", ".join(parts)


@dataclass
class SipMessage:
    start_line: str
    headers: dict[str, str]
    record_route: list[str] = field(default_factory=list)
    body: str = ""
    raw_headers: list[str] = field(default_factory=list)

    @property
    def status(self) -> int | None:
        if not self.start_line.startswith("SIP/2.0 "):
            return None
        return int(self.start_line.split()[1])

    @property
    def to_tag(self) -> str:
        return re.search(r"tag=([^;>]+)", self.headers["to"]).group(1)


def parse(raw: str) -> SipMessage:
    head, _, body = raw.partition("\r\n\r\n")
    lines = head.split("\r\n")
    headers: dict[str, str] = {}
    record_route: list[str] = []
    for line in lines[1:]:
        name, _, value = line.partition(":")
        name = name.strip().lower()
        if name == "record-route":
            record_route.extend(v.strip() for v in value.split(","))
        headers.setdefault(name, value.strip())
    return SipMessage(lines[0], headers, record_route, body, lines[1:])


@dataclass
class Dialog:
    """What a confirmed INVITE leaves behind for the ACK and the BYE."""

    call_id: str
    to_uri: str
    to_tag: str
    remote_target: str
    route: str
    answer_sdp: str


class SipClient:
    """One endpoint's signalling over an open WebSocket (subprotocol ``sip``)."""

    def __init__(self, ws, user: str, password: str) -> None:
        self._ws = ws
        self._user = user
        self._password = password
        self._host = f"{secrets.token_hex(6)}.invalid"
        self._from_tag = secrets.token_hex(5)
        self._cseq = 0

    def _build(
        self,
        method: str,
        uri: str,
        to: str,
        call_id: str,
        *,
        cseq: int,
        to_tag: str = "",
        extra: tuple[str, ...] = (),
        body: str = "",
    ) -> str:
        lines = [
            f"{method} {uri} SIP/2.0",
            f"Via: SIP/2.0/WSS {self._host};branch=z9hG4bK{secrets.token_hex(8)}",
            "Max-Forwards: 70",
            f"To: <{to}>" + (f";tag={to_tag}" if to_tag else ""),
            f"From: <sip:{self._user}@{DOMAIN}>;tag={self._from_tag}",
            f"Call-ID: {call_id}",
            f"CSeq: {cseq} {method}",
            f"Contact: <sip:{self._user}@{self._host};transport=ws;ob>",
            "Allow: INVITE,ACK,CANCEL,BYE,UPDATE,MESSAGE,OPTIONS,REFER,INFO,NOTIFY",
            "Supported: ice,replaces,outbound",
            "User-Agent: rory-plivo",
            *extra,
        ]
        if body:
            lines.append("Content-Type: application/sdp")
        lines.append(f"Content-Length: {len(body.encode())}")
        return "\r\n".join(lines) + "\r\n\r\n" + body

    async def _transaction(self, method: str, uri: str, to: str, call_id: str, extra=(), body="") -> SipMessage:
        """Send a request, answer one digest challenge, return the final response."""
        self._cseq += 1
        await self._ws.send(self._build(method, uri, to, call_id, cseq=self._cseq, extra=extra, body=body))
        challenged = False
        while True:
            msg = parse(await self._ws.recv())
            status = msg.status
            if status is None or status < 200:
                continue
            if status in (401, 407) and not challenged:
                challenged = True
                if method == "INVITE":
                    # A non-2xx final response to INVITE is ACKed hop by hop.
                    await self._ws.send(self._build("ACK", uri, to, call_id, cseq=self._cseq, to_tag=msg.to_tag))
                challenge = msg.headers.get("www-authenticate") or msg.headers["proxy-authenticate"]
                header = "Authorization" if status == 401 else "Proxy-Authorization"
                auth = f"{header}: {digest_authorization(method, uri, self._user, self._password, challenge)}"
                self._cseq += 1
                await self._ws.send(
                    self._build(method, uri, to, call_id, cseq=self._cseq, extra=(auth, *extra), body=body)
                )
                continue
            return msg

    async def register(self) -> None:
        aor = f"sip:{self._user}@{DOMAIN}"
        msg = await self._transaction("REGISTER", f"sip:{DOMAIN}", aor, secrets.token_hex(10), extra=("Expires: 600",))
        if msg.status != 200:
            raise RuntimeError(f"Plivo REGISTER failed: {msg.start_line}")

    async def invite(self, destination: str, offer_sdp: str) -> Dialog:
        """INVITE, then ACK the 200 OK along its route set. Returns the confirmed dialog."""
        to_uri = f"sip:{destination}@{DOMAIN}"
        call_id = secrets.token_hex(10)
        msg = await self._transaction("INVITE", to_uri, to_uri, call_id, body=offer_sdp)
        if msg.status != 200:
            raise RuntimeError(f"Plivo INVITE failed: {msg.start_line}")
        dialog = Dialog(
            call_id=call_id,
            to_uri=to_uri,
            to_tag=msg.to_tag,
            remote_target=re.search(r"<([^>]+)>", msg.headers["contact"]).group(1),
            route="Route: " + ", ".join(reversed(msg.record_route)),
            answer_sdp=msg.body,
        )
        await self._ws.send(
            self._build(
                "ACK", dialog.remote_target, to_uri, call_id,
                cseq=self._cseq, to_tag=dialog.to_tag, extra=(dialog.route,),
            )
        )
        return dialog

    async def wait_for_remote_bye(self) -> None:
        """Answer in-dialog requests with 200 OK until Plivo sends BYE (the call ended on its side)."""
        while True:
            msg = parse(await self._ws.recv())
            if msg.status is not None:
                continue  # a late response, e.g. a retransmitted 200 OK
            echoed = [h for h in msg.raw_headers if h.split(":", 1)[0].strip().lower() in ("via", "from", "to", "call-id", "cseq")]
            await self._ws.send("\r\n".join(["SIP/2.0 200 OK", *echoed, "Content-Length: 0"]) + "\r\n\r\n")
            if msg.start_line.startswith("BYE "):
                return

    async def bye(self, dialog: Dialog) -> None:
        self._cseq += 1
        await self._ws.send(
            self._build(
                "BYE", dialog.remote_target, dialog.to_uri, dialog.call_id,
                cseq=self._cseq, to_tag=dialog.to_tag, extra=(dialog.route,),
            )
        )
