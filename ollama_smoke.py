#!/usr/bin/env python3
"""Bounded smoke check for an Ollama model's structured output."""

from __future__ import annotations

import argparse
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "expected_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "expected_points"],
    "additionalProperties": False,
}
PROMPT = (
    "You are testing an interview coach. Ask one interview question about a "
    "synthetic note stating: 'A cache stores reusable results and invalidation "
    "removes stale entries.' Return the question and expected answer points."
)
METRIC_FIELDS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)
MAX_RESPONSE_BYTES = 256 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _api_url(endpoint: str) -> str:
    if not isinstance(endpoint, str):
        raise ValueError("endpoint must be an explicit loopback URL")
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint must be an explicit http://127.0.0.1:PORT or http://[::1]:PORT")
    return endpoint.rstrip("/") + "/api/chat"


def _structured(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ValueError("response content is not a JSON string")
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"question", "expected_points"}:
        raise ValueError("response does not match the required object fields")
    if not isinstance(value["question"], str) or not value["question"].strip():
        raise ValueError("question must be a non-empty string")
    points = value["expected_points"]
    if (
        not isinstance(points, list)
        or not points
        or not all(isinstance(point, str) and point.strip() for point in points)
    ):
        raise ValueError("expected_points must be a non-empty string array")
    return value


def _response_json(response: Any) -> Any:
    data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("ollama response exceeds 256 KiB")
    return json.loads(data)


def smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Run the smoke check and return a JSON-shaped diagnostic result."""
    model = config.get("model", "qwen3.5:9b")
    started = time.perf_counter()
    result: dict[str, Any] = {
        "success": False,
        "model": model,
        "elapsed_ms": 0,
        "metrics": {key: None for key in METRIC_FIELDS},
    }
    try:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        url = _api_url(config["endpoint"])
        timeout = float(config.get("timeout", 120))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "format": SCHEMA,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 256},
            }
        ).encode()
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            payload = _response_json(response)
        if not isinstance(payload, dict):
            raise ValueError("server response is not a JSON object")
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ValueError("server response has no message object")
        result["metrics"].update(
            {key: payload[key] for key in METRIC_FIELDS if isinstance(payload.get(key), int) and payload[key] >= 0}
        )
        _structured(message.get("content"))
        result["success"] = True
    except KeyError:
        result["error"] = "endpoint is required"
    except urllib.error.HTTPError as error:
        result["error"] = "model unavailable" if error.code == 404 else f"ollama returned HTTP {error.code}"
    except TimeoutError:
        result["error"] = "ollama request timed out"
    except urllib.error.URLError as error:
        result["error"] = (
            "ollama request timed out" if isinstance(error.reason, TimeoutError) else "ollama is unavailable"
        )
    except OSError:
        result["error"] = "ollama is unavailable"
    except json.JSONDecodeError:
        result["error"] = "model returned invalid structured JSON"
    except (ValueError, TypeError, UnicodeError) as error:
        result["error"] = str(error)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def self_check() -> None:
    assert _api_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434/api/chat"
    assert _api_url("http://[::1]:11434/") == "http://[::1]:11434/api/chat"
    for unsafe in ("http://localhost:11434", "https://127.0.0.1:11434", "http://8.8.8.8:11434"):
        try:
            _api_url(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe endpoint: {unsafe}")
    assert _structured('{"question":"Why invalidate?","expected_points":["stale data"]}')
    assert _response_json(io.BytesIO(b'{"ok":true}')) == {"ok": True}
    try:
        _response_json(io.BytesIO(b" " * (MAX_RESPONSE_BYTES + 1)))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted oversized response")
    for invalid in ('{"question":""}', '{"question":"Q","expected_points":[]}'):
        try:
            _structured(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid structured output: {invalid}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="explicit Ollama loopback URL, including port")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print(json.dumps({"success": True, "self_check": True}))
        return 0
    result = smoke(vars(args))
    print(json.dumps(result, separators=(",", ":")))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
