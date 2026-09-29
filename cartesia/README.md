# rory-cartesia

Rory on Cartesia Managed Agents. Transport only — the tools, vendor clients and
prompt live in rory-core.

Cartesia hosts the voice loop: Ink speech recognition, the LLM (`gpt-4.1`),
Sonic speech and turn-taking. At startup the pod creates Rory's 16 tools as
webhook tools, once. For each call it:

1. creates a managed agent on them — the shared prompt plus the frozen-clock
   date, the shared greeting as `initial_message`;
2. opens the agent's WebSocket at the actor's `pcm_24000` and relays audio both
   ways;
3. answers the webhooks on `POST /tool/<tool name>`;
4. deletes the agent when the call ends.

The tools are deleted when the pod shuts down.

## Tool budget: 16 per pod, 100 per account

A Cartesia account may store at most 100 tools. One pod holds 16, so at most
**6 pods** can run on one account at a time; keep parallel calls at or below
that. A pod that cannot create its tools deletes the ones it did create and
fails its health check with the vendor's error, e.g.
`HTTP 400 {"error_code":"tool_limit_reached", ...}`. Creating the tools per
call instead would exhaust the account after six concurrent calls.

## Why Managed Agents, not the Line SDK

- Cartesia stops hosting Line SDK agents on December 1, 2026; Managed Agents
  replace them.
- `cartesia-line` pins `websockets<14`, `starlette<1.0` and `fastapi<0.135.2`.
  The shared lockfile cannot hold those beside the transports that run
  websockets 15.
- A Line agent id routes calls to one registered URL at a time, so attempts could
  not run in parallel. Here every pod creates its own tools and every call its
  own agent, and tool names need not be unique in an account, so parallel
  attempts don't collide.

## Tool calls arrive over HTTPS

Cartesia POSTs the model's arguments to the tool's URL, so the pod needs a
public HTTPS address routed to `:8008`, given as `PUBLIC_BASE_URL`. The boot
fails without an `https://` value.

A tool's URL and bearer are fixed when it is created, so both are per pod: the
URL is `PUBLIC_BASE_URL/tool/<tool name>` and the bearer is a secret minted once
per process, which Cartesia stores write-only on the tools. The webhook answers
401 for a wrong bearer and 404 when the pod is not on a call, and otherwise runs
the tool through `rory_tools.dispatch` under the call's lock.

Nothing in a webhook names the call, so a pod is on one call at a time and
refuses a second concurrent `/voice` connection. Run one pod, with its own
public endpoint, per concurrent call.

## Vendor limits

- The LLM catalog has no `gpt-4.1-mini`, the model the other cascades use;
  `gpt-4.1` is the nearest.
- The webhook body schema has no `minimum` keyword and rejects a tool that
  carries one, so the adapter drops it from `amount_cents`. The parity test pins
  that this is the only difference from the shared declaration.
- Webhook responses are truncated after 4 KiB. Rory's largest observed result,
  `get_account`, is about 1.6 KiB.

## Trace evidence

The model runs on Cartesia's side and tools arrive by webhook, so
OpenTelemetry instrumentation in the pod sees no model calls. The agent
WebSocket's `turn_ended` events do carry `tool_calls`, and the bridge logs
them.

## Running it

```
docker build --platform linux/amd64 -f cartesia/Dockerfile -t rory-cartesia .
```

The image serves `/voice`, `/health` and `/tool/<tool name>` on `:8008`; set
`PUBLIC_BASE_URL` to its public HTTPS address.

## Environment

| Variable | Default | Role |
|---|---|---|
| `CARTESIA_API_KEY` | — | Required. Creates the agent and tools, opens the WebSocket. |
| `PUBLIC_BASE_URL` | — | Required, `https://`. The pod's public endpoint. |
| `CARTESIA_MODEL` | `gpt-4.1` | LLM id from `GET /v1/agents/models`. |
| `CARTESIA_VOICE_ID` | `db6b0ed5-d5d3-463d-ae85-518a07d3c2b4` | Sonic voice ("Skylar – Friendly Guide"). |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run --package rory-cartesia pytest -q cartesia/tests
```

`test_parity_cartesia.py` runs the shared parity assertions (allowing only the
dropped `minimum`), pins the webhook tool shape and checks the container fails
closed at boot without its key or public URL. `test_lifecycle_cartesia.py` runs
two calls against a fake Cartesia: the 16 tools are created once and shared by
both agents, deleted at shutdown after the agents, a full account fails the
boot and cleans up, and the webhook checks the bearer before the live call.
