#!/usr/bin/env python3
"""Bounded smoke check for an Ollama model's structured output."""

from __future__ import annotations

import argparse
import http.client
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
MAX_REQUEST_BYTES = 8192


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
        or not 1 <= parsed.port <= 65535
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


def _invalid_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _response_json(response: Any) -> Any:
    data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("ollama response exceeds 256 KiB")
    return json.loads(data, parse_constant=_invalid_constant, parse_float=_finite_float)


def chat(request: dict[str, Any]) -> dict[str, Any]:
    """Send one bounded local request; the controller validates proposal fields."""
    started = time.perf_counter()
    result: dict[str, Any] = {
        "ok": False,
        "content": None,
        "error": "invalid request",
        "metrics": {key: None for key in METRIC_FIELDS},
    }
    try:
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        request_limit = request.get("max_request_bytes", MAX_REQUEST_BYTES)
        if type(request_limit) is not int or not 0 < request_limit <= MAX_REQUEST_BYTES:
            raise ValueError("request limit must be positive and at most 8192 bytes")
        model = request.get("model", "qwen3.5:9b")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        url = _api_url(request["endpoint"])
        timeout = float(request.get("timeout", 120))
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("timeout must be positive and at most 120 seconds")
        messages = request["messages"]
        schema = request["format"]
        if (
            not isinstance(messages, list)
            or not messages
            or not all(
                isinstance(message, dict)
                and set(message) == {"role", "content"}
                and message["role"] in ("system", "user", "assistant")
                and isinstance(message["content"], str)
                and message["content"].strip()
                for message in messages
            )
            or not isinstance(schema, dict)
            or not schema
        ):
            raise ValueError("messages and JSON schema are required")
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 256},
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > request_limit:
            result["error"] = f"request exceeds {request_limit} bytes"
            return result
        http_request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        result["error"] = "model returned invalid structured JSON"
        with opener.open(http_request, timeout=timeout) as response:
            payload = _response_json(response)
        if not isinstance(payload, dict) or payload.get("done") is not True or payload.get("done_reason") == "length":
            raise ValueError("server response is incomplete")
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ValueError("server response has no message object")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("response content is not a JSON string")
        content = json.loads(content, parse_constant=_invalid_constant, parse_float=_finite_float)
        if not isinstance(content, dict):
            raise ValueError("response content is not a JSON object")
        result["metrics"].update(
            {key: payload[key] for key in METRIC_FIELDS if type(payload.get(key)) is int and payload[key] >= 0}
        )
        result.update(ok=True, content=content, error=None)
    except KeyError as error:
        result["error"] = "endpoint is required" if error.args == ("endpoint",) else "messages and format are required"
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
    except http.client.HTTPException:
        result["error"] = "model returned invalid structured JSON"
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as error:
        if str(error) == "ollama response exceeds 256 KiB":
            result["error"] = "ollama response exceeds 256 KiB"
    finally:
        result["metrics"]["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Run the smoke check and return a JSON-shaped diagnostic result."""
    response = chat({**config, "messages": [{"role": "user", "content": PROMPT}], "format": SCHEMA})
    result: dict[str, Any] = {
        "success": response["ok"],
        "model": config.get("model", "qwen3.5:9b"),
        "elapsed_ms": response["metrics"]["elapsed_ms"],
        "metrics": {key: response["metrics"][key] for key in METRIC_FIELDS},
    }
    if response["ok"]:
        try:
            _structured(json.dumps(response["content"]))
        except ValueError:
            result.update(success=False, error="model returned invalid structured JSON")
    else:
        result["error"] = response["error"]
    return result


def self_check() -> None:
    from unittest.mock import Mock, patch

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
    request = {
        "endpoint": "http://127.0.0.1:11434",
        "messages": [{"role": "user", "content": "Synthetic cache note."}],
        "format": SCHEMA,
    }
    valid = {"question": "Why invalidate?", "expected_points": ["stale data"]}
    payload = {"done": True, "message": {"content": json.dumps(valid)}, "eval_count": 4}
    opener = Mock()
    with patch("urllib.request.build_opener", return_value=opener) as build:
        opener.open.return_value = io.BytesIO(json.dumps(payload).encode())
        result = chat({**request, "stream": True, "think": True, "options": {"num_ctx": 99999}})
        assert result["ok"] and result["content"] == valid and result["error"] is None
        assert result["metrics"]["eval_count"] == 4
        sent = opener.open.call_args.args[0]
        body = json.loads(sent.data)
        assert body["messages"] == request["messages"] and body["format"] == SCHEMA
        assert body["options"] == {"temperature": 0, "num_ctx": 4096, "num_predict": 256}
        assert body["stream"] is False and body["think"] is False
        assert opener.open.call_args.kwargs == {"timeout": 120}
        proxy, redirect = build.call_args.args
        assert isinstance(proxy, urllib.request.ProxyHandler) and proxy.proxies == {}
        assert isinstance(redirect, NoRedirect)
        assert redirect.redirect_request(None, None, 302, "", {}, "https://example.com") is None
        for data in (
            b"private malformed content",
            b" " * (MAX_RESPONSE_BYTES + 1),
            json.dumps({"done": False, "message": payload["message"]}).encode(),
            json.dumps({**payload, "done_reason": "length"}).encode(),
            json.dumps({"done": True, "message": {"content": "[1]"}}).encode(),
            json.dumps({"done": True, "message": {"content": '{"x":NaN}'}}).encode(),
            json.dumps({"done": True, "message": {"content": '{"x":1e999}'}}).encode(),
            b'{"done":true,"message":{"content":"{}"},"total_duration":1e999}',
        ):
            opener.open.return_value = io.BytesIO(data)
            result = chat(request)
            assert result["ok"] is False and result["content"] is None
            assert "private" not in result["error"]
        for error, classification in (
            (urllib.error.HTTPError(sent.full_url, 404, "private error", {}, None), "model unavailable"),
            (urllib.error.HTTPError(sent.full_url, 302, "redirect", {}, None), "ollama returned HTTP 302"),
            (TimeoutError(), "ollama request timed out"),
            (urllib.error.URLError("private error"), "ollama is unavailable"),
            (http.client.IncompleteRead(b"private partial body"), "model returned invalid structured JSON"),
        ):
            opener.open.side_effect = error
            result = chat(request)
            assert result["error"] == classification and result["content"] is None
        opener.open.side_effect = None
        for change in (
            {"timeout": 121},
            {"timeout": float("nan")},
            {"messages": []},
            {"endpoint": "http://localhost:11434"},
            {"max_request_bytes": MAX_REQUEST_BYTES + 1},
            {"max_request_bytes": 0},
            {"max_request_bytes": True},
            {"max_request_bytes": 3072.0},
        ):
            opener.open.reset_mock()
            assert chat({**request, **change})["ok"] is False
            opener.open.assert_not_called()
        # Include every serialized field, and count UTF-8 bytes rather than characters.
        request["messages"][0]["content"] = ""
        body["messages"] = request["messages"]
        overhead = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())
        for limit in (MAX_REQUEST_BYTES, 3072):
            bounded = {**request, **({"max_request_bytes": limit} if limit == 3072 else {})}
            request["messages"][0]["content"] = "\u00e9" + "x" * (limit - overhead - 2)
            opener.open.return_value = io.BytesIO(json.dumps(payload).encode())
            assert chat(bounded)["ok"] is True
            assert len(opener.open.call_args.args[0].data) == limit
            request["messages"][0]["content"] += "x"
            opener.open.reset_mock()
            assert chat(bounded)["error"] == f"request exceeds {limit} bytes"
            opener.open.assert_not_called()
        opener.open.return_value = io.BytesIO(json.dumps(payload).encode())
        assert smoke({"endpoint": request["endpoint"]})["success"] is True
        opener.open.return_value = io.BytesIO(json.dumps({"done": True, "message": {"content": '{}'}}).encode())
        assert smoke({"endpoint": request["endpoint"]})["success"] is False


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
