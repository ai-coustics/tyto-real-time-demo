"""What can this key actually use?

    uv run tools/list_models.py                       # OPENAI_API_KEY from .env
    uv run tools/list_models.py --key sk-...          # a specific key
    uv run tools/list_models.py --env OPENAI_API_KEY_STAGING
    uv run tools/list_models.py --probe               # confirm access, do not trust the list
    uv run tools/list_models.py --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 --env INKLING_API_KEY

``GET /v1/models`` lists what an organisation can see, which is not the same as
what a key may call. This repo already has a case in point: the Tinker endpoint
answers that call with an empty list while happily serving
``thinkingmachines/Inkling-Small``, and the OpenAI Realtime API accepts any model
name in ``session.update`` and only fails later, per item. So ``--probe`` sends a
one-token completion to each candidate and reports what came back. That is the
only answer worth trusting.

Works against any OpenAI-compatible endpoint, so the same tool covers OpenAI and
Tinker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tyto_voice.env import load_env  # noqa: E402

OPENAI_BASE = "https://api.openai.com/v1"
# Some endpoints reject urllib's default agent outright; see inkling.py.
USER_AGENT = "tyto-voice/0.1 (+https://github.com/ai-coustics)"
TIMEOUT = 30.0

# Chat models worth probing when the listing is unhelpful or absent. Kept broad
# on purpose: a 404 here is information, not a failure.
CANDIDATES = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5",
    "o4-mini",
)


def _request(url: str, key: str, payload: dict | None = None) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        body = err.read().decode(errors="replace")
        try:
            return err.code, json.loads(body)
        except Exception:  # noqa: BLE001
            return err.code, body[:200]
    except Exception as err:  # noqa: BLE001
        return 0, str(err)[:200]


def list_models(base_url: str, key: str) -> list[str]:
    status, body = _request(f"{base_url.rstrip('/')}/models", key)
    if status != 200 or not isinstance(body, dict):
        print(f"  GET /models -> {status}: {str(body)[:160]}")
        return []
    ids = sorted(m.get("id", "") for m in body.get("data", []))
    if not ids:
        print("  GET /models -> 200 but the list is empty. That does not mean the key is "
              "unusable; Tinker answers this way. Use --probe.")
    return [i for i in ids if i]


def probe(base_url: str, key: str, models: list[str]) -> None:
    print(f"  probing {len(models)} model(s) with a one-token call:")
    for model in models:
        t0 = time.time()
        status, body = _request(
            f"{base_url.rstrip('/')}/chat/completions",
            key,
            {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        )
        dt = time.time() - t0
        if status == 200:
            print(f"    OK    {model:34s} {dt:5.2f}s")
        else:
            detail = body.get("error", {}).get("message") if isinstance(body, dict) else body
            detail = detail or (body.get("detail") if isinstance(body, dict) else "")
            print(f"    {status:<5} {model:34s} {str(detail)[:80]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", help="the key itself, instead of reading it from the environment")
    parser.add_argument("--env", default="OPENAI_API_KEY", help="environment variable holding the key")
    parser.add_argument("--base-url", default=OPENAI_BASE)
    parser.add_argument("--probe", action="store_true",
                        help="send a one-token call per model, the only reliable check")
    parser.add_argument("--only", nargs="*", help="probe just these model ids")
    args = parser.parse_args()

    load_env()
    key = args.key or os.environ.get(args.env, "")
    if not key:
        print(f"No key. {args.env} is unset or empty, and --key was not given.")
        return 1
    print(f"endpoint : {args.base_url}")
    print(f"key      : {args.env} ({key[:7]}...{key[-4:]}, {len(key)} chars)")

    listed = list_models(args.base_url, key)
    if listed:
        chat = [m for m in listed if not any(
            t in m for t in ("embedding", "whisper", "tts", "dall-e", "moderation", "audio"))]
        print(f"  {len(listed)} model(s) listed, {len(chat)} look like chat models:")
        for m in chat:
            print(f"    {m}")

    if args.probe or args.only:
        targets = args.only or [m for m in (listed or CANDIDATES) if m in (listed or CANDIDATES)]
        probe(args.base_url, key, targets or list(CANDIDATES))
    else:
        print("  (add --probe to confirm the key can actually call them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
