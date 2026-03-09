"""
Entry point for the web agent. Validator runs: uvicorn main:app --host 0.0.0.0 --port $SANDBOX_AGENT_PORT
"""
from agent import app

__all__ = ["app"]
