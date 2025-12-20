# Deployment Readiness Plan

> Scope: Fix all audit findings (critical/major/minor), align README vs reality, enforce settings, and harden runtime for production deployment.  
> Rule: Complete items in order; validate each step with targeted checks.  
> Status legend: [ ] pending, [x] done.

## 0) Safety + Baseline
- [ ] Snapshot current state (git status) and identify any unrelated changes.
- [ ] Confirm API keys are local only; ensure `.env` not committed.
- [ ] Ensure local DB is backed up before any schema changes.

## 1) Critical Fixes (must be done before anything else)
### 1.1 Trailing Stop Logic (NO-side bug)
- [ ] Fix NO-side profit calculation and trailing stop update direction.
- [ ] Add unit-level tests for YES/NO trailing stops (activation/trigger).
- [ ] Verify trailing stops update only in favorable direction.

### 1.2 Risk Manager order params (Kalshi requires exactly one price field)
- [ ] Fix risk reduction sell orders to set only one of `yes_price` or `no_price`.
- [ ] Apply same fix to partial sell path.
- [ ] Add tests for order param serialization.

### 1.3 Performance jobs syntax errors
- [ ] Fix indentation and undefined variables in `src/jobs/evaluate.py`.
- [ ] Fix indentation in `src/jobs/performance_analyzer.py`.
- [ ] Add quick smoke tests to run both jobs without crashing.

## 2) Major Reliability + Correctness
### 2.1 Decision volatility math
- [ ] Fix `estimate_market_volatility` to avoid double-dividing prices.
- [ ] Add a regression test using a 0.5 price case.

### 2.2 CLI strategy toggles
- [ ] Ensure CLI flags update `settings.trading.*` not module-level globals.
- [ ] Verify `--no-market-making` and `--directional-only` actually disable strategy.

### 2.3 Client lifecycle and leaks
- [ ] Reuse or properly close Kalshi/XAI clients created in `run_trading_job`.
- [ ] Ensure `beast_mode_bot.py` cycle uses shared clients when possible.
- [ ] Add async teardown on shutdown.

### 2.4 Unified AI budget enforcement
- [ ] Define a single source of truth for daily AI spend (DB or tracker).
- [ ] Record all AI costs through the same system.
- [ ] Ensure `beast_mode_bot` and `decide.py` enforce the same limits.

## 3) Settings Compliance + Consistency
### 3.1 Remove or wire unused settings
- [ ] Wire `max_daily_loss_pct` into risk manager or remove it.
- [ ] Fix `profit_threshold`/`loss_threshold` to be used in tracking order placement.
- [ ] Audit any remaining unused settings and either wire or delete.

### 3.2 Unit consistency (percent vs decimal)
- [ ] Normalize cash reserve and limit percentages (0–100 vs 0–1).
- [ ] Update code to consistently treat all percentages the same way.
- [ ] Update README + `env.template` comments to match units.

### 3.3 Module-level settings cleanup
- [ ] Remove or fully synchronize module-level duplicates in `settings.py`.
- [ ] Ensure `Settings.__post_init__` no longer creates drift.

## 4) Strategy/Feature Hardening
### 4.1 Market Making
- [ ] Implement real order refresh/cancel/re-quote logic.
- [ ] Ensure spread logic handles stale data safely.
- [ ] Add basic order tracking to measure fill performance.

### 4.2 Quick Flip Scalping
- [ ] Ensure sell order placement checks for actual fills (avoid placing exits for failed entries).
- [ ] Add tests for time-based price adjustments and 30‑min cut.

### 4.3 Portfolio Optimization
- [ ] Validate Kelly math vs expected formula; add unit test with known values.
- [ ] Document caps/volatility adjustments and ensure they’re intentional.

### 4.4 Arbitrage
- [ ] Decide whether to enable non‑zero allocation.
- [ ] If enabled, add a liquidity gate + leg-hedge verification.

## 5) README vs Reality Alignment
- [ ] Update README claims to match actual implementations (multi-agent prompt, ML heuristics, risk parity approximation, dashboard polling).
- [ ] Clearly mark optional/disabled features (arbitrage allocation, options strategies).
- [ ] Add explicit notes on strategy allocation overrides (dynamic allocation).

## 6) Testing + Validation
- [ ] Add/extend pytest coverage for:
  - trailing stop YES/NO
  - risk manager order params
  - volatility estimation
  - Kelly sizing
  - strategy allocation enforcement
- [ ] Run `pytest` and fix any regressions.

## 7) Deployment Readiness Checks
- [ ] Verify `requirements.txt` contains all runtime dependencies.
- [ ] Confirm Docker image runs with `start_railway.sh` end‑to‑end.
- [ ] Validate dashboard polling works under production settings.
- [ ] Confirm reconciliation job runs daily and writes cache.

## 8) Operational Monitoring
- [ ] Add logging for AI budget usage per cycle.
- [ ] Log order failures with structured metadata.
- [ ] Surface health alerts on dashboard (critical issues + cost warnings).

## 9) Final Release Checklist
- [ ] Re-run full audit checklist (critical/major/minor). 
- [ ] Confirm README claims are accurate.
- [ ] Confirm settings table is fully compliant.
- [ ] Confirm no secrets in repo or logs.
- [ ] Optional: create release notes.
