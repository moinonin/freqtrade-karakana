# freqtrade-karakana

Standalone Docker runtime and Compose stack for **Freqtrade + Karakana** (Structural Geometry Optimization), published as [`moinonin/freqtrade-karakana`](https://hub.docker.com/r/moinonin/freqtrade-karakana) (`linux/amd64` and `linux/arm64`).

Freqtrade itself is pulled from [`freqtradeorg/freqtrade:2026.3`](https://hub.docker.com/r/freqtradeorg/freqtrade); this image layers the `karakana` Python package, the two shadow/health sidecar CLIs (`karakana-shadow-trader`, `karakana-shadow-health`), and operational calibration tools on top.

## Quick Start

### Prerequisites
- Docker Engine with **Docker Compose v2** (`docker compose version`).
  - On Ubuntu/Debian if `docker compose` is missing:
    ```sh
    sudo apt-get update && sudo apt-get install -y docker-compose-plugin
    ```

### 1. Clone and prepare `user_data/`

```sh
git clone https://github.com/moinonin/freqtrade-karakana.git
cd freqtrade-karakana

mkdir -p user_data/{strategies,karakana_policy/watch,logs,reports}
cp config/config.example.json user_data/config.json
```

Place your strategy and trained Karakana policy artifacts into `user_data/` (kept out of the image and git so keys, DBs, and weights stay private):

```text
user_data/
├── config.json                        # copied from config/config.example.json (dry_run: true by default)
├── strategies/
│   └── <YourStrategy>.py              # e.g. Strategy322sgo.py (matches FT_STRATEGY)
└── karakana_policy/
    └── watch/
        ├── best_policy.pt             # trained OfflineTrading policy (v0 3-action or v1 2-action scale_invariant_v1)
        └── offline_preprocessing.json # matching normalization contract from the same training run
```

> **Note:** `config/config.example.json` defaults to `dry_run: true` on Bybit futures with empty API keys. Only add exchange credentials when you are ready for live trading, and never commit `user_data/config.json`.

### 2. Optional `.env` overrides

Create a `.env` file alongside `docker-compose-cloud.yml` if you want to override defaults:

```dotenv
KARAKANA_IMAGE=moinonin/freqtrade-karakana:1.0.7
FT_STRATEGY=Strategy322sgo
FT_CONFIG_FILE=config.json
FT_DB_FILE=tradesv3.sqlite

# Fast-start arming (arms after 8 shadow trades; raise for stricter warmup)
KARAKANA_HISTORICAL_MIN_ARMED_ROWS=8
KARAKANA_HISTORICAL_MIN_ARMED_LONG_COUNT=2
KARAKANA_HISTORICAL_MIN_ARMED_SHORT_COUNT=2

# Set to 1 to block entries unless shadow reward_mean > 0 after fees
KARAKANA_HISTORICAL_REQUIRE_POSITIVE_EV=0

# Pessimistic round-trip fee assumption (bps) for shadow trades & threshold sweeps
KARAKANA_SHADOW_FEE_BPS=5
KARAKANA_THRESHOLD_SCHEDULER_FEE_BPS=5
```

### 3. Pull and run the 4-service stack

```sh
docker compose -f docker-compose-cloud.yml pull
docker compose -f docker-compose-cloud.yml up -d
docker compose -f docker-compose-cloud.yml ps
```

The stack launches four isolated containers sharing `./user_data`:
1. **`freqtrade-karakana`** — runs Freqtrade, exports live signal rows, and reads dynamic thresholds from the health CSV.
2. **`karakana-shadow-trader`** — replays live signals into virtual fee-adjusted shadow trades (`user_data/reports/karakana_shadow_trade_history.csv`).
3. **`karakana-shadow-health`** — evaluates directional SGO health on shadow trades and publishes dynamic thresholds (`user_data/reports/directional_strategy_health.csv`).
4. **`karakana-threshold-scheduler`** — runs periodic fee-aware threshold sweeps (requiring positive expectancy after fees) and compacts append-only CSVs.

### 4. Verify

```sh
# Smoke-test the image (note: base image sets ENTRYPOINT ["freqtrade"])
docker run --rm --entrypoint /bin/sh moinonin/freqtrade-karakana:1.0.7 -c '
  python -c "from karakana.runtime.policy import KarakanaOfflinePolicy, SCALE_INVARIANT_V1; print(\"OK:\", SCALE_INVARIANT_V1)" &&
  python /freqtrade/karakana_tools/sweep_live_shadow_thresholds.py --help >/dev/null && echo "SWEEP OK" &&
  python /freqtrade/karakana_tools/threshold_scheduler.py --help >/dev/null && echo "SCHEDULER OK"
'

# Check stack logs and artifact growth
docker compose -f docker-compose-cloud.yml ps
tail -n 20 user_data/logs/freqtrade.log
wc -l user_data/freqtrade_signals_karakana.csv \
      user_data/reports/karakana_shadow_trade_history.csv \
      user_data/reports/directional_strategy_health.csv
```

## What's in this repo

| Path | Role |
|---|---|
| `Dockerfile` | Two-stage multi-arch build layering `karakana` and `tools/` onto `freqtradeorg/freqtrade:2026.3` |
| `docker-compose-cloud.yml` | Pull-only 4-service Compose stack for any server/VPS |
| `config/config.example.json` | Sanitized Freqtrade dry-run config template (no secrets) |
| `config/karakana-config.json` | Default Karakana SGO profile config (copied to `/freqtrade/karakana/`) |
| `pyproject.toml` | Karakana package metadata (console scripts: `karakana-shadow-trader`, `karakana-shadow-health`) |
| `karakana/` | Python package (SGO math, trainers, and runtime policy/shadow/health sidecars) |
| `tools/` | Operational scripts baked into `/freqtrade/karakana_tools/`: `threshold_scheduler.py`, `sweep_live_shadow_thresholds.py`, `monitor_karakana_runtime.py` |
| `Makefile` | Local buildx targets (`make build`, `make push`, `make load-arm64`, `make inspect`) |
| `.github/workflows/docker.yml` | CI: builds and pushes `linux/amd64` + `linux/arm64` on `v*` tags or manual dispatch |
| `OPERATIONS.md` | Detailed operations, one-off debugging commands, version history, and troubleshooting |

### Pinned runtime dependencies

- `torch==2.11.0+cpu` (PyTorch CPU wheel index)
- `numpy==2.2.6`, `pandas==2.3.3`
- `ccxt==4.5.54` (with `setuptools<82`)
- `gymnasium==1.2.3`
- `ft-scikit-optimize==0.9.2`, `termcolor`, `pygments`

## Build & publish

```sh
# Local multi-arch build (cached, not pushed)
make build

# Build + push linux/amd64 and linux/arm64, retag :latest
make push TAG=1.0.7

# Arm64-only image loaded into local docker for smoke tests
make load-arm64

# Inspect pushed manifest and per-platform digests
make inspect TAG=1.0.7
```

CI publishing: push a tag `v1.0.7` (or trigger the workflow manually). Requires repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

## Known quirk

`karakana/config.py` resolves `config/karakana-config.json` relative to the CWD (`/freqtrade/config/...`) or the package parent — neither exists in the image, so `load_config()` silently falls back to its built-in defaults. The config copied to `/freqtrade/karakana/` is therefore inert unless `KARAKANA_CONFIG=/freqtrade/karakana/karakana-config.json` is set. Put runtime overrides in `user_data/` or `.env`, not in the image.
