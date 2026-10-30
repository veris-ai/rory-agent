"""Provision the AWS side of the AgentCore candidate, end to end. Safe to re-run.

    uv run --package rory-agentcore python agentcore/deploy.py

Run from the repo root with AWS credentials for the target account. In order:

1. DynamoDB table for per-call session records
2. tools Lambda (Rory's 16 tools through the shared dispatcher)
3. AgentCore Gateway (MCP, IAM auth) with the tools Lambda as its target
4. AgentCore Runtime (the agent container, built for arm64 and pushed to ECR)
5. init Lambda (writes a call's session record and warms its AgentCore session)
6. Lex hook Lambda, and a Lex V2 bot that only transcribes, with a ``live`` alias
7. IAM user the bench pod uses for Lex, Polly, the init Lambda and the runtime, and nothing else

Prints the values the pod needs. Re-running updates code and configuration in place;
the Lex bot is built once.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import boto3
from rory_agentcore.contract import GATEWAY_TARGET
from rory_agentcore.tools import GATEWAY_TOOLS

REGION = "us-east-1"
PREFIX = "rory-agentcore"
TABLE = f"{PREFIX}-sessions"
RUNTIME_NAME = "rory_agentcore"  # runtime names allow no hyphens
MODEL_ID = "us.anthropic.claude-sonnet-5-5"
LOCALE = "en_US"
ROOT = Path(__file__).resolve().parent.parent

session = boto3.Session(region_name=REGION)
ACCOUNT = session.client("sts").get_caller_identity()["Account"]
iam = session.client("iam")
ddb = session.client("dynamodb")
lam = session.client("lambda")
ecr = session.client("ecr")
acc = session.client("bedrock-agentcore-control")
lex = session.client("lexv2-models")


def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def wait_status(what: str, fetch, key: str, ready: str, failed: tuple[str, ...] = (), timeout_s: int = 900) -> None:
    """Poll ``fetch()[key]`` until it reads ``ready``; a ``failed`` status raises with the full description."""
    deadline = time.monotonic() + timeout_s
    while True:
        described = fetch()
        status = described[key]
        if status == ready:
            return
        if status in failed:
            raise RuntimeError(f"{what} {status}: {described}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{what} still {status}")
        time.sleep(5)


# --------------------------------------------------------------------------- IAM


def role(name: str, service: str, policy: dict | None, managed: list[str] = ()) -> str:
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": service},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT}},
        }],
    }
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=json.dumps(trust))
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        log(f"created role {name}")
    if policy is not None:
        iam.put_role_policy(RoleName=name, PolicyName=f"{name}-inline", PolicyDocument=json.dumps(policy))
    for policy_arn in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=policy_arn)
    return arn


def statement(actions: list[str], resources: list[str]) -> dict:
    return {"Effect": "Allow", "Action": actions, "Resource": resources}


def policy(*statements: dict) -> dict:
    return {"Version": "2012-10-17", "Statement": list(statements)}


LAMBDA_BASIC = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

# ---------------------------------------------------------------------- DynamoDB


def table() -> str:
    try:
        arn = ddb.describe_table(TableName=TABLE)["Table"]["TableArn"]
    except ddb.exceptions.ResourceNotFoundException:
        arn = ddb.create_table(
            TableName=TABLE,
            AttributeDefinitions=[{"AttributeName": "session_id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )["TableDescription"]["TableArn"]
        ddb.get_waiter("table_exists").wait(TableName=TABLE)
        ddb.update_time_to_live(
            TableName=TABLE, TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"}
        )
        log(f"created table {TABLE}")
    return arn


# ------------------------------------------------------------------------ Lambda


def lambda_zip() -> bytes:
    """rory_tools and this package, with rory-core's locked dependencies for Lambda's Linux."""
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp) / "pkg"
        reqs = Path(tmp) / "requirements.txt"
        subprocess.run(
            ["uv", "export", "--frozen", "--package", "rory-core", "--no-dev", "--no-emit-workspace",
             "--no-hashes", "--no-header", "-o", str(reqs)],
            cwd=ROOT, check=True, capture_output=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "-r", str(reqs), "--target", str(build), "--python-platform",
             "x86_64-manylinux2014", "--python-version", "3.12", "--only-binary", ":all:", "--quiet"],
            cwd=ROOT, check=True,
        )
        shutil.copytree(ROOT / "rory_tools", build / "rory_tools", ignore=shutil.ignore_patterns("__pycache__", "data"))
        pkg = build / "rory_agentcore"
        pkg.mkdir()
        for name in ("__init__.py", "contract.py", "tools.py", "tools_lambda.py", "init_lambda.py", "lex_hook.py"):
            shutil.copy(ROOT / "agentcore" / "rory_agentcore" / name, pkg / name)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(build.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    zf.write(path, path.relative_to(build))
        return buf.getvalue()


def function(name: str, role_arn: str, handler: str, code: bytes, env: dict[str, str], timeout: int) -> str:
    config = {
        "FunctionName": name, "Role": role_arn, "Handler": handler, "Runtime": "python3.12",
        # Lambda's CPU scales with memory; TLS to the twins and JSON want the CPU.
        "Timeout": timeout, "MemorySize": 1024, "Environment": {"Variables": env},
    }
    try:
        lam.get_function(FunctionName=name)
        lam.update_function_code(FunctionName=name, ZipFile=code)
        lam.get_waiter("function_updated_v2").wait(FunctionName=name)
        arn = lam.update_function_configuration(**config)["FunctionArn"]
        lam.get_waiter("function_updated_v2").wait(FunctionName=name)
        log(f"updated function {name}")
    except lam.exceptions.ResourceNotFoundException:
        # A just-created role takes a few seconds before Lambda may assume it.
        for attempt in range(12):
            try:
                arn = lam.create_function(**config, Code={"ZipFile": code})["FunctionArn"]
                break
            except lam.exceptions.InvalidParameterValueException as exc:
                if "cannot be assumed" not in str(exc) or attempt == 11:
                    raise
                time.sleep(5)
        lam.get_waiter("function_active_v2").wait(FunctionName=name)
        log(f"created function {name}")
    return arn


# ----------------------------------------------------------------------- Gateway


def gateway(tools_lambda_arn: str) -> tuple[str, str]:
    """The MCP gateway in front of the tools Lambda: (gateway ARN, MCP URL)."""
    role_arn = role(
        f"{PREFIX}-gateway", "bedrock-agentcore.amazonaws.com",
        policy(statement(["lambda:InvokeFunction"], [tools_lambda_arn])),
    )
    existing = [g for g in acc.list_gateways()["items"] if g["name"] == PREFIX]
    if existing:
        gateway_id = existing[0]["gatewayId"]
    else:
        for attempt in range(12):
            try:
                gateway_id = acc.create_gateway(
                    name=PREFIX, roleArn=role_arn, protocolType="MCP", authorizerType="AWS_IAM",
                )["gatewayId"]
                break
            except acc.exceptions.ValidationException as exc:
                if attempt == 11:
                    raise
                log(f"gateway create retry: {exc}")
                time.sleep(5)
        log(f"created gateway {gateway_id}")
    wait_status("gateway", lambda: acc.get_gateway(gatewayIdentifier=gateway_id), "status", "READY", ("FAILED",))

    target = {"mcp": {"lambda": {"lambdaArn": tools_lambda_arn, "toolSchema": {"inlinePayload": GATEWAY_TOOLS}}}}
    credentials = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
    targets = [t for t in acc.list_gateway_targets(gatewayIdentifier=gateway_id)["items"] if t["name"] == GATEWAY_TARGET]
    if targets:
        target_id = targets[0]["targetId"]
        acc.update_gateway_target(
            gatewayIdentifier=gateway_id, targetId=target_id, name=GATEWAY_TARGET,
            targetConfiguration=target, credentialProviderConfigurations=credentials,
        )
    else:
        target_id = acc.create_gateway_target(
            gatewayIdentifier=gateway_id, name=GATEWAY_TARGET,
            targetConfiguration=target, credentialProviderConfigurations=credentials,
        )["targetId"]
        log(f"created gateway target {target_id}")
    wait_status(
        "gateway target", lambda: acc.get_gateway_target(gatewayIdentifier=gateway_id, targetId=target_id),
        "status", "READY", ("FAILED",),
    )
    described = acc.get_gateway(gatewayIdentifier=gateway_id)
    return described["gatewayArn"], described["gatewayUrl"]


# ----------------------------------------------------------------------- Runtime


def runtime_image() -> str:
    """Build the agent container for arm64, push it to ECR, and return it by digest."""
    repo = f"{PREFIX}-runtime"
    try:
        uri = ecr.describe_repositories(repositoryNames=[repo])["repositories"][0]["repositoryUri"]
    except ecr.exceptions.RepositoryNotFoundException:
        uri = ecr.create_repository(repositoryName=repo)["repository"]["repositoryUri"]
        log(f"created ECR repository {repo}")
    token = ecr.get_authorization_token()["authorizationData"][0]
    password = base64.b64decode(token["authorizationToken"]).decode().split(":", 1)[1]
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", token["proxyEndpoint"]],
        input=password.encode(), check=True, capture_output=True,
    )
    tag = f"{uri}:latest"
    subprocess.run(
        ["docker", "buildx", "build", "--platform", "linux/arm64", "-f", "agentcore/runtime.Dockerfile",
         "-t", tag, "--push", "."],
        cwd=ROOT, check=True,
    )
    digest = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", tag, "--format", "{{json .Manifest}}"],
        check=True, capture_output=True, text=True,
    ).stdout
    return f"{uri}@{json.loads(digest)['digest']}"


def runtime(gateway_arn: str, gateway_url: str) -> str:
    repo_arn = f"arn:aws:ecr:{REGION}:{ACCOUNT}:repository/{PREFIX}-runtime"
    role_arn = role(
        f"{PREFIX}-runtime", "bedrock-agentcore.amazonaws.com",
        policy(
            statement(["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], ["*"]),
            statement(["bedrock-agentcore:InvokeGateway"], [gateway_arn]),
            statement(["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"], [repo_arn]),
            statement(["ecr:GetAuthorizationToken"], ["*"]),
            statement(
                ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
                 "logs:DescribeLogGroups"],
                [f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/bedrock-agentcore/runtimes/*",
                 f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:*"],
            ),
        ),
    )
    image = runtime_image()
    config = {
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": image}},
        "roleArn": role_arn,
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "protocolConfiguration": {"serverProtocol": "HTTP"},
        "environmentVariables": {"GATEWAY_URL": gateway_url, "BEDROCK_MODEL_ID": MODEL_ID},
    }
    existing = [r for r in acc.list_agent_runtimes()["agentRuntimes"] if r["agentRuntimeName"] == RUNTIME_NAME]
    if existing:
        runtime_id = existing[0]["agentRuntimeId"]
        acc.update_agent_runtime(agentRuntimeId=runtime_id, **config)
        log(f"updated runtime {runtime_id} to {image}")
    else:
        for attempt in range(12):
            try:
                runtime_id = acc.create_agent_runtime(agentRuntimeName=RUNTIME_NAME, **config)["agentRuntimeId"]
                break
            except acc.exceptions.ValidationException as exc:
                if attempt == 11:
                    raise
                log(f"runtime create retry: {exc}")
                time.sleep(5)
        log(f"created runtime {runtime_id}")
    wait_status(
        "runtime", lambda: acc.get_agent_runtime(agentRuntimeId=runtime_id), "status", "READY",
        ("CREATE_FAILED", "UPDATE_FAILED"),
    )
    return acc.get_agent_runtime(agentRuntimeId=runtime_id)["agentRuntimeArn"]


# --------------------------------------------------------------------------- Lex


def lex_role() -> str:
    name = "AWSServiceRoleForLexV2Bots_RoryAgentcore"
    try:
        return iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_service_linked_role(
            AWSServiceName="lexv2.amazonaws.com", CustomSuffix="RoryAgentcore"
        )["Role"]["Arn"]
        log("created Lex service-linked role")
        time.sleep(10)
        return arn


def bot(hook_arn: str) -> tuple[str, str]:
    """A Lex bot that only transcribes: every utterance falls to the fallback intent,
    whose hook asks for the caller's next turn and nothing else.

    The pod reads the transcript off the conversation stream and takes it to the
    agent itself, so the reply can stream back as it is generated. The hook keeps
    the dialog open; without it Lex closes it after every utterance.
    """
    bots = [b for b in lex.list_bots()["botSummaries"] if b["botName"] == PREFIX]
    if bots:
        bot_id = bots[0]["botId"]
    else:
        bot_id = lex.create_bot(
            botName=PREFIX, roleArn=lex_role(), dataPrivacy={"childDirected": False},
            idleSessionTTLInSeconds=900,
        )["botId"]
        wait_status("bot", lambda: lex.describe_bot(botId=bot_id), "botStatus", "Available", ("Failed",))
        lex.create_bot_locale(botId=bot_id, botVersion="DRAFT", localeId=LOCALE, nluIntentConfidenceThreshold=0.4)
        locale = lambda: lex.describe_bot_locale(botId=bot_id, botVersion="DRAFT", localeId=LOCALE)
        wait_status("locale", locale, "botLocaleStatus", "NotBuilt", ("Failed",))
        # Lex will not build a locale without one intent of its own. This one is never
        # meant to match; everything the caller says goes to the fallback.
        lex.create_intent(
            botId=bot_id, botVersion="DRAFT", localeId=LOCALE, intentName="Unused",
            sampleUtterances=[{"utterance": "rory agentcore placeholder intent never spoken"}],
        )
        fallback = next(i for i in lex.list_intents(botId=bot_id, botVersion="DRAFT", localeId=LOCALE)["intentSummaries"]
                        if i.get("parentIntentSignature") == "AMAZON.FallbackIntent")
        lex.update_intent(
            botId=bot_id, botVersion="DRAFT", localeId=LOCALE, intentId=fallback["intentId"],
            intentName=fallback["intentName"], parentIntentSignature="AMAZON.FallbackIntent",
            fulfillmentCodeHook={"enabled": True},
        )
        lex.build_bot_locale(botId=bot_id, botVersion="DRAFT", localeId=LOCALE)
        wait_status("locale build", locale, "botLocaleStatus", "Built", ("Failed",))
        log(f"created bot {bot_id}")

    versions = [v for v in lex.list_bot_versions(botId=bot_id)["botVersionSummaries"] if v["botVersion"] != "DRAFT"]
    if versions:
        version = versions[0]["botVersion"]
    else:
        version = lex.create_bot_version(
            botId=bot_id, botVersionLocaleSpecification={LOCALE: {"sourceBotVersion": "DRAFT"}}
        )["botVersion"]

    def describe_version():
        # A new version is not describable for a few seconds after it is created.
        try:
            return lex.describe_bot_version(botId=bot_id, botVersion=version)
        except lex.exceptions.ResourceNotFoundException:
            return {"botStatus": "Creating"}

    wait_status("bot version", describe_version, "botStatus", "Available", ("Failed",))

    locale_settings = {LOCALE: {"enabled": True, "codeHookSpecification": {
        "lambdaCodeHook": {"lambdaARN": hook_arn, "codeHookInterfaceVersion": "1.0"}}}}
    aliases = [a for a in lex.list_bot_aliases(botId=bot_id)["botAliasSummaries"] if a["botAliasName"] == "live"]
    if aliases:
        alias_id = aliases[0]["botAliasId"]
        lex.update_bot_alias(botId=bot_id, botAliasId=alias_id, botAliasName="live", botVersion=version,
                             botAliasLocaleSettings=locale_settings)
    else:
        alias_id = lex.create_bot_alias(botId=bot_id, botAliasName="live", botVersion=version,
                                        botAliasLocaleSettings=locale_settings)["botAliasId"]
        log(f"created alias {alias_id}")
    wait_status("alias", lambda: lex.describe_bot_alias(botId=bot_id, botAliasId=alias_id), "botAliasStatus",
                "Available", ("Failed",))

    alias_arn = f"arn:aws:lex:{REGION}:{ACCOUNT}:bot-alias/{bot_id}/{alias_id}"
    try:
        lam.add_permission(
            FunctionName=hook_arn, StatementId=f"lex-{bot_id}-{alias_id}", Action="lambda:InvokeFunction",
            Principal="lexv2.amazonaws.com", SourceArn=alias_arn,
        )
    except lam.exceptions.ResourceConflictException:
        pass  # granted on an earlier run
    return bot_id, alias_id


# ---------------------------------------------------------------------- Bench user


def bench_user(bot_id: str, alias_id: str, init_arn: str, runtime_arn: str) -> str:
    """The IAM user the bench pod signs in as: Lex on this alias, Polly, the init
    Lambda, and the agent runtime."""
    name = f"{PREFIX}-bench"
    try:
        iam.get_user(UserName=name)
    except iam.exceptions.NoSuchEntityException:
        iam.create_user(UserName=name)
        log(f"created user {name}")
    iam.put_user_policy(UserName=name, PolicyName=f"{name}-inline", PolicyDocument=json.dumps(policy(
        statement(["lex:StartConversation"], [f"arn:aws:lex:{REGION}:{ACCOUNT}:bot-alias/{bot_id}/{alias_id}"]),
        statement(["polly:SynthesizeSpeech"], ["*"]),
        statement(["lambda:InvokeFunction"], [init_arn]),
        statement(["bedrock-agentcore:InvokeAgentRuntime"], [runtime_arn, f"{runtime_arn}/*"]),
    )))
    return name


# -------------------------------------------------------------------------- main


def main() -> None:
    table_arn = table()
    code = lambda_zip()
    log(f"lambda package {len(code) / 1e6:.1f} MB")

    tools_role = role(
        f"{PREFIX}-tools-lambda", "lambda.amazonaws.com",
        policy(statement(["dynamodb:GetItem", "dynamodb:UpdateItem"], [table_arn])), [LAMBDA_BASIC],
    )
    tools_arn = function(f"{PREFIX}-tools", tools_role, "rory_agentcore.tools_lambda.handler", code,
                         {"SESSIONS_TABLE": TABLE}, timeout=60)

    gateway_arn, gateway_url = gateway(tools_arn)
    runtime_arn = runtime(gateway_arn, gateway_url)

    init_role = role(
        f"{PREFIX}-init-lambda", "lambda.amazonaws.com",
        policy(
            statement(["dynamodb:PutItem"], [table_arn]),
            statement(["bedrock-agentcore:InvokeAgentRuntime"], [runtime_arn, f"{runtime_arn}/*"]),
        ),
        [LAMBDA_BASIC],
    )
    init_arn = function(f"{PREFIX}-init", init_role, "rory_agentcore.init_lambda.handler", code,
                        {"SESSIONS_TABLE": TABLE, "AGENTCORE_RUNTIME_ARN": runtime_arn}, timeout=30)

    hook_role = role(f"{PREFIX}-lex-hook-lambda", "lambda.amazonaws.com", None, [LAMBDA_BASIC])
    hook_arn = function(f"{PREFIX}-lex-hook", hook_role, "rory_agentcore.lex_hook.handler", code, {}, timeout=5)
    bot_id, alias_id = bot(hook_arn)
    user = bench_user(bot_id, alias_id, init_arn, runtime_arn)

    print(json.dumps({
        "AWS_REGION": REGION,
        "LEX_BOT_ID": bot_id,
        "LEX_BOT_ALIAS_ID": alias_id,
        "INIT_LAMBDA": f"{PREFIX}-init",
        "AGENTCORE_RUNTIME_ARN": runtime_arn,
        "GATEWAY_URL": gateway_url,
        "BENCH_IAM_USER": user,
    }, indent=2))


if __name__ == "__main__":
    sys.exit(main())
