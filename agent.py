"""
Web Agent for Subnet 36. Exposes /health and POST /act.
Uses LLM to decide the next browser action based on task, DOM, and history.
"""
from __future__ import annotations

import json
import os
import re
from html import unescape
from typing import Any

from bs4 import BeautifulSoup
from fastapi import Body, FastAPI

from llm_gateway import chat_completions

app = FastAPI(title="Subnet 36 Web Agent")

# Truncate HTML to avoid token limits; keep structure for selectors
MAX_HTML_CHARS = 12000

SYSTEM_PROMPT = """You are a web automation agent. Given a task, current URL, and HTML snapshot, you must decide the NEXT SINGLE action to perform.

Available action types (return exactly ONE):
1. navigate - go to a URL: {"type": "navigate", "url": "https://..."}
2. click - click an element: {"type": "click", "selector": "css-selector"}
3. input - type into an input: {"type": "input", "selector": "css-selector", "value": "text to type"}

Rules:
- Return ONLY a valid JSON object for ONE action. No markdown, no explanation.
- Use precise CSS selectors from the HTML (id, unique classes, button text via :contains if supported, or a[href="..."]).
- For inputs, use input[name="..."], input#id, or similar. Avoid generic div/span unless unique.
- If the task seems complete or no clear action, return: {"type": "stop"}
- For links: a[href="/path"] or a:has-text("Link Text") if your evaluator supports it. Fallback: use the exact href.

Output format (single line JSON): {"type": "...", "selector": "...", "value": "..."} or {"type": "navigate", "url": "..."}
"""


def _truncate_html(html: str, max_chars: int = MAX_HTML_CHARS) -> str:
    """Truncate HTML while keeping useful structure."""
    if not html:
        return ""
    html = html.strip()
    if len(html) <= max_chars:
        return html
    # Try to cut at a tag boundary
    cut = html[:max_chars]
    last_open = cut.rfind("<")
    last_close = cut.rfind(">")
    if last_close > last_open and last_close >= max_chars - 50:
        return cut[: last_close + 1] + "\n<!-- truncated -->"
    return cut + "\n<!-- truncated -->"


def _simplify_html(html: str) -> str:
    """Remove script/style and collapse whitespace for smaller context."""
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.prettify()
        text = re.sub(r"\s+", " ", text)
        return unescape(text).strip()
    except Exception:
        return html[:MAX_HTML_CHARS]


def _extract_action_from_llm_response(content: str) -> dict[str, Any] | None:
    """Parse LLM response into a single action dict."""
    content = (content or "").strip()
    # Handle markdown code blocks
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)
    # Find first JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            action_type = (obj.get("type") or "").strip().lower()
            if action_type == "stop":
                return None
            if action_type == "navigate":
                url = (obj.get("url") or "").strip()
                if url:
                    return {"type": "navigate", "url": url}
            elif action_type == "click":
                sel = (obj.get("selector") or "").strip()
                if sel:
                    return {"type": "click", "selector": sel}
            elif action_type == "input":
                sel = (obj.get("selector") or "").strip()
                val = obj.get("value", "")
                if sel:
                    return {"type": "input", "selector": sel, "value": str(val)}
        return None
    except json.JSONDecodeError:
        return None


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/act", summary="Decide next action")
async def act(payload: dict[str, Any] = Body(...)) -> dict[str, list[dict[str, Any]]]:
    """
    Receive task context and return one action. Validator executes one action per call.
    """
    # Support flat or nested payload (task object)
    task_obj = payload.get("task") or payload
    task_id = str((payload.get("task_id") or task_obj.get("task_id") or task_obj.get("id") or "")).strip() or "unknown"
    prompt = str(payload.get("prompt") or task_obj.get("prompt") or "").strip()
    url = str(payload.get("url") or task_obj.get("url") or payload.get("current_url") or "").strip()
    snapshot_html = str(payload.get("snapshot_html") or task_obj.get("snapshot_html") or payload.get("html") or "").strip()
    step_index = int(payload.get("step_index", 0))
    history = payload.get("history") or []

    # Simplify and truncate HTML
    html_snippet = _simplify_html(snapshot_html)
    html_snippet = _truncate_html(html_snippet)

    history_text = ""
    if history:
        steps = []
        for h in history[-5:]:  # Last 5 steps
            a = h.get("action") or h.get("type") or "?"
            err = h.get("error")
            ok = h.get("exec_ok", True)
            steps.append(f"  step: {a} ok={ok}" + (f" error={err}" if err else ""))
        history_text = "\n".join(steps)

    user_content = f"""Task: {prompt}
Current URL: {url}
Step: {step_index}

Previous steps:
{history_text or "  (none)"}

Current page HTML (excerpt):
{html_snippet}

What is the NEXT single action? Return only a JSON object."""

    try:
        body = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }
        response = chat_completions(
            task_id=task_id,
            body=body,
            timeout_seconds=25.0,
        )
        choices = response.get("choices") or []
        if not choices:
            return {"actions": []}
        content = (choices[0].get("message") or {}).get("content") or ""
        action = _extract_action_from_llm_response(content)
        if action:
            return {"actions": [action]}
    except Exception as e:
        # Log and return no action to avoid crashes
        import traceback
        traceback.print_exc()
        # Fallback: try to navigate to task URL if we haven't yet
        if step_index == 0 and url and not history:
            return {"actions": [{"type": "navigate", "url": url}]}

    return {"actions": []}


@app.post("/step", summary="Alias for /act")
async def step(payload: dict[str, Any] = Body(...)) -> dict[str, list[dict[str, Any]]]:
    return await act(payload)
