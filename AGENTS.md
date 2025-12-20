# Repository Guidelines

## Project Structure & Module Organization
- `src/` houses the core trading system, organized by domain: `clients/`, `jobs/`, `strategies/`, `utils/`, `analytics/`, and `config/`.
- Entry-point scripts live at the repo root (for example, `beast_mode_bot.py`, `launch_dashboard.py`, `init_database.py`).
- Tests are in `tests/` with `test_*.py` files; fixtures live under `tests/fixtures/`.
- Local data and runtime artifacts may appear in `logs/`, `trading_bot.db`, and `trading_system.db`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` creates and activates a local virtual env.
- `pip install -r requirements.txt` installs runtime deps; `pip install -r requirements-dev.txt` installs test deps.
- `python init_database.py` initializes the SQLite schema.
- `python beast_mode_bot.py` runs paper trading; add `--live` for real trading and `--dashboard` for the UI.
- `python launch_dashboard.py` starts the monitoring dashboard.
- `python run_tests.py` opens the interactive test menu; `pytest` runs the full test suite.

## Coding Style & Naming Conventions
- Python style follows PEP 8 with 4-space indentation and 88-char lines.
- Use `snake_case` for functions/vars, `PascalCase` for classes, and clear module names in `src/`.
- Preferred tooling (per `CONTRIBUTING.md`): `black`, `isort`, and `mypy`. Install them if you plan to format or type-check.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` for async tests.
- Keep test names descriptive and start files with `test_`.
- Aim for >80% coverage when adding significant logic; mock external APIs (Kalshi/xAI) in tests.
- Example: `pytest tests/test_decide.py::test_make_decision_for_market_creates_position -v -s`.

## Commit & Pull Request Guidelines
- Prefer conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`. History is mixed, but follow the guideline for new work.
- PRs should include a brief description, testing notes, and a checklist (see `CONTRIBUTING.md` for the template).
- Never include API keys, secrets, or real trading credentials in commits or PRs.

## Security & Configuration Tips
- Copy `env.template` to `.env` and keep it local; do not commit `.env`.
- Keep `LIVE_TRADING_ENABLED=false` unless intentionally testing with real funds.
