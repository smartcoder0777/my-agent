#!/usr/bin/env python3
from __future__ import annotations

"""Run eval.py multiple times and aggregate summaries.

This is a generic utility for miners comparing configs/models.
The run key format is `provider:model` and is passed via env vars:
- LLM_PROVIDER
- OPENAI_MODEL
"""

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunSpec:
    provider: str
    model: str

    @property
    def slug(self) -> str:
        s = f"{self.provider}__{self.model}".lower()
        return re.sub(r"[^a-z0-9._-]+", "_", s)


def _parse_run(raw: str) -> RunSpec:
    if ":" not in raw:
        raise SystemExit(f"Invalid run '{raw}'. Use provider:model")
    provider, model = raw.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise SystemExit(f"Invalid run '{raw}'")
    return RunSpec(provider=provider, model=model)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare eval runs across provider:model configs"
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        action="append",
        required=True,
        help="provider:model entries",
    )
    ap.add_argument("--agent-base-url", default="http://127.0.0.1:5000")
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--num-tasks", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout-seconds", type=float, default=30.0)
    args = ap.parse_args()

    run_specs: list[RunSpec] = []
    for group in args.runs or []:
        for item in group:
            run_specs.append(_parse_run(item))

    out_dir = REPO_DIR / "data" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for spec in run_specs:
        run_out = out_dir / f"{spec.slug}.json"
        env = os.environ.copy()
        env["LLM_PROVIDER"] = spec.provider
        env["OPENAI_MODEL"] = spec.model

        cmd = [
            env.get("PYTHON", "python"),
            str(REPO_DIR / "eval.py"),
            "--agent-base-url",
            str(args.agent_base_url),
            "--num-tasks",
            str(int(args.num_tasks)),
            "--repeat",
            str(int(args.repeat)),
            "--timeout-seconds",
            str(float(args.timeout_seconds)),
            "--out",
            str(run_out),
        ]
        if args.tasks_file:
            cmd.extend(["--tasks-file", str(args.tasks_file)])

        print(f"\n=== RUN {spec.provider}:{spec.model} ===")
        proc = subprocess.run(cmd, cwd=str(REPO_DIR), env=env, check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

        payload = json.loads(run_out.read_text(encoding="utf-8"))
        s = payload.get("summary") or {}
        summary_rows.append(
            {
                "provider": spec.provider,
                "model": spec.model,
                "ok_calls": int(s.get("ok_calls") or 0),
                "total_calls": int(s.get("total_calls") or 0),
                "ok_rate": float(s.get("ok_rate") or 0.0),
                "avg_latency_ms": float(s.get("avg_latency_ms") or 0.0),
                "out": str(run_out),
            }
        )

    summary_rows.sort(key=lambda r: (-r["ok_rate"], r["avg_latency_ms"]))
    summary = {"runs": summary_rows}
    summary_path = out_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Compare Summary ===")
    for row in summary_rows:
        print(
            f"{row['provider']:10s} {row['model']:30.30s} "
            f"{row['ok_calls']}/{row['total_calls']} ({row['ok_rate']:.1%}) "
            f"avg={row['avg_latency_ms']:.1f}ms"
        )
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
