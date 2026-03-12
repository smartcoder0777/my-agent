"""
OpenAI-compatible LLM gateway with IWA-Task-ID header for sandbox cost tracking.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
IWA_TASK_ID_HEADER = "IWA-Task-ID"


def is_sandbox_gateway_base_url(base_url: str) -> bool:
    """Return True when base_url looks like the local validator gateway."""
    try:
        url = httpx.URL((base_url or "").strip())
        host = (url.host or "").lower()
        return host in {"sandbox-gateway", "localhost", "127.0.0.1"}
    except Exception:
        return False


def gateway_headers(*, task_id: str, api_key: str | None = None) -> dict[str, str]:
    """Build OpenAI-compatible headers with mandatory IWA task correlation."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        IWA_TASK_ID_HEADER: str(task_id),
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def chat_completions(
    *,
    task_id: str,
    body: dict[str, Any],
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Call OpenAI-compatible /chat/completions through a gateway."""
    resolved_base_url = (
        base_url or os.getenv(OPENAI_BASE_URL_ENV, "https://api.openai.com/v1")
    ).rstrip("/")
    resolved_api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")

    if not resolved_api_key and not is_sandbox_gateway_base_url(resolved_base_url):
        raise RuntimeError(
            "OPENAI_API_KEY not set and OPENAI_BASE_URL is not a local sandbox gateway"
        )

    headers = gateway_headers(task_id=task_id, api_key=resolved_api_key or None)
    url = f"{resolved_base_url}/chat/completions"

    with httpx.Client(timeout=float(timeout_seconds)) as client:
        resp = client.post(url, json=body, headers=headers)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            snippet = (exc.response.text or "")[:300]
            raise RuntimeError(
                f"chat/completions failed ({exc.response.status_code}): {snippet}"
            ) from exc
        return resp.json()


def openai_chat_completions(
    *,
    task_id: str,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 350,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Convenience wrapper matching the reference agent's calling convention."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return chat_completions(task_id=task_id, body=body, timeout_seconds=timeout_seconds)
