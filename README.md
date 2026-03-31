# Subnet 36 Web Agent

LLM-powered web agent for Bittensor Subnet 36. Exposes `GET /health` and `POST /act` as required by validators.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure (copy and add OPENAI_API_KEY)
cp .env.example .env

# Run
uvicorn main:app --host 0.0.0.0 --port 5000
```

## API Contract

- **GET /health** - Returns `{"status": "healthy"}`
- **POST /act** - Accepts task/HTML/history, returns `{"actions": [{"type": "...", ...}]}`

## Environment

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (or Chutes etc. via gateway) |
| `OPENAI_BASE_URL` | Override API URL (sandbox sets this to gateway) |
| `OPENAI_MODEL` | Model name (default: gpt-4o-mini) |

## Miner Setup

1. Push this repo to GitHub
2. In `autoppia_web_agents_subnet/.env` set:
   ```
   GITHUB_URL="https://github.com/yourusername/my_agent/commit/abc123..."
   AGENT_NAME="My Agent"
   ```
3. Start the miner with PM2

## Supported Actions

- `navigate` - `{"type": "navigate", "url": "https://..."}`
- `click` - `{"type": "click", "selector": "#submit-btn"}`
- `input` - `{"type": "input", "selector": "#email", "value": "text"}`

## Local evaluation (`eval_github`)

The subnet repo (`autoppia_web_agents_subnet`) runs `python -m scripts.miner.eval_github --github ...` against your agent. Before that, the **IWA demo backend** must be listening on **port 8090** (task generation and DB reset).

1. Clone the demo webs repo alongside the subnet repo (default path expected by the deploy script):
   ```bash
   cd ~/work
   git clone https://github.com/autoppia/autoppia_webs_demo.git
   ```
2. From `autoppia_web_agents_subnet`, run the deploy script (or set `WEBS_DEMO_PATH` to your clone):
   ```bash
   ./scripts/validator/demo-webs/deploy_demo_webs.sh
   ```
3. Confirm the API is up:
   ```bash
   curl -sSf http://localhost:8090/health
   ```
4. Install **Playwright** browsers and host deps on Linux (`playwright install`, `sudo playwright install-deps` if prompted).
5. Run the miner eval from the subnet repo root with your API keys in an env file:
   ```bash
   python -m scripts.miner.eval_github \
     --env-file /path/to/.env \
     --github "https://github.com/YOUR_USER/YOUR_REPO/commit/YOUR_SHA" \
     --tasks 1
   ```

Install `autoppia_iwa` exactly as the subnet repo documents (same commit/venv as validators) so task generation does not hit dependency-injector / LLM wiring errors.

**URL handling:** Demo apps use non-default ports (e.g. `http://localhost:8013/?seed=...`). This agent resolves navigation URLs against the current page and **realigns** `http://localhost/...` (implicit port 80) to the same host/port as the task URL so the browser does not open the wrong origin.
