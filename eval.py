#!/usr/bin/env python3
from __future__ import annotations

"""Generic `/act` evaluator for miner templates.

This is intentionally tool-only and does not contain agentic logic.
It validates endpoint availability, response shape, and timing.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp


REPO_DIR = Path(__file__).resolve().parent


def _default_tasks(num_tasks: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for i in range(max(1, int(num_tasks))):
        tasks.append(
            {
                "task_id": f"template-task-{i}",
                "prompt": "Template task for /act schema validation",
                "url": "https://example.com",
                "snapshot_html": "<html><body><button>Continue</button></body></html>",
                "step_index": i,
                "history": [],
            }
        )
    return tasks


def _load_tasks(path: str | None, fallback_num_tasks: int) -> list[dict[str, Any]]:
    if not path:
        return _default_tasks(fallback_num_tasks)

    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        data = raw["tasks"]
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError("tasks file must be a JSON list or {'tasks':[...]} object")

    tasks: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        tasks.append(
            {
                "task_id": str(item.get("task_id") or f"task-{i}"),
                "prompt": str(item.get("prompt") or "Template task"),
                "url": str(item.get("url") or "https://example.com"),
                "snapshot_html": str(item.get("snapshot_html") or "<html></html>"),
                "step_index": int(item.get("step_index") or 0),
                "history": item.get("history")
                if isinstance(item.get("history"), list)
                else [],
            }
        )
    return tasks


def _validate_actions_shape(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, f"response is not an object ({type(payload).__name__})"
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return False, "missing or invalid 'actions' list"
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            return False, f"actions[{idx}] is not an object"
        t = action.get("type")
        if t is not None and not isinstance(t, str):
            return False, f"actions[{idx}].type must be a string when present"
    return True, "ok"


async def _call_act(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    async with session.post(f"{base_url.rstrip('/')}/act", json=payload) as resp:
        body = await resp.json(content_type=None)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "status": int(resp.status),
            "elapsed_ms": elapsed_ms,
            "body": body,
        }


async def run_eval(
    *,
    agent_base_url: str,
    tasks: list[dict[str, Any]],
    repeat: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []

    timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for r in range(max(1, int(repeat))):
            for idx, task in enumerate(tasks):
                result: dict[str, Any] = {
                    "repeat_index": r,
                    "task_index": idx,
                    "task_id": task.get("task_id"),
                    "ok": False,
                }
                try:
                    call = await _call_act(
                        session, base_url=agent_base_url, payload=task
                    )
                    result["status"] = call["status"]
                    result["elapsed_ms"] = call["elapsed_ms"]

                    shape_ok, shape_msg = _validate_actions_shape(call["body"])
                    result["shape_ok"] = shape_ok
                    result["shape_msg"] = shape_msg

                    actions = (
                        call["body"].get("actions")
                        if isinstance(call["body"], dict)
                        else []
                    )
                    result["action_count"] = (
                        len(actions) if isinstance(actions, list) else 0
                    )
                    result["ok"] = (int(call["status"]) == 200) and bool(shape_ok)
                except Exception as exc:
                    result["error"] = str(exc)
                episodes.append(result)

    ok_count = sum(1 for ep in episodes if ep.get("ok"))
    avg_latency = (
        (sum(int(ep.get("elapsed_ms") or 0) for ep in episodes) / len(episodes))
        if episodes
        else 0.0
    )

    return {
        "agent_base_url": agent_base_url,
        "num_tasks": len(tasks),
        "repeat": int(repeat),
        "episodes": episodes,
        "summary": {
            "total_calls": len(episodes),
            "ok_calls": ok_count,
            "ok_rate": (ok_count / len(episodes)) if episodes else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generic /act evaluator for miner templates"
    )
    ap.add_argument("--agent-base-url", default="http://127.0.0.1:5000")
    ap.add_argument(
        "--tasks-file",
        default=None,
        help="Optional JSON file: list[...] or {'tasks':[...]} ",
    )
    ap.add_argument("--num-tasks", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout-seconds", type=float, default=30.0)
    ap.add_argument("--out", default=str(REPO_DIR / "data" / "eval_result.json"))
    args = ap.parse_args()

    tasks = _load_tasks(args.tasks_file, args.num_tasks)
    result = asyncio.run(
        run_eval(
            agent_base_url=str(args.agent_base_url),
            tasks=tasks,
            repeat=int(args.repeat),
            timeout_seconds=float(args.timeout_seconds),
        )
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = result["summary"]
    print("=== Eval Summary ===")
    print(f"calls:      {summary['ok_calls']}/{summary['total_calls']} ok")
    print(f"ok_rate:    {summary['ok_rate']:.1%}")
    print(f"avg_latency:{summary['avg_latency_ms']} ms")
    print(f"out:        {out_path}")


if __name__ == "__main__":
    main()
