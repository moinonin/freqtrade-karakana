# freqtrade-karakana Image — Operations Guide

This document covers the published Docker image `moinonin/freqtrade-karakana` and how to operate the Freqtrade + Karakana stack.

**Image:** `moinonin/freqtrade-karakana:1.0.7` (multi-arch: `linux/amd64`, `linux/arm64`)
**Base:** `freqtradeorg/freqtrade:2026.3` + Karakana runtime + sidecar CLIs

---

## What's In The Image

| Path | Contents |
|------|----------|
| `/freqtrade/karakana_tools/threshold_scheduler.py` | Periodic sweep scheduler |
| `/freqtrade/karakana_tools/sweep_live_shadow_thresholds.py` | Grid sweep + compose patcher |
| `/freqtrade/karakana_tools/monitor_karakana_runtime.py` | CSV compaction + status JSON |
| `/usr/local/bin/karakana-shadow-trader` | Console script (virtual trades) |
| `/usr/local/bin/karakana-shadow-health` | Console script (dynamic thresholds) |
| `/freqtrade/karakana/karakana-config.json` | Default config (inert unless `KARAKANA_CONFIG` set) |

**Pinned deps:** `torch==2.11.0+cpu`, `numpy==2.2.6`, `pandas==2.3.3`, `ccxt==4.5.54`, `gymnasium==1.2.3`

---

## Quick Start (Pull & Run)

```bash
# 1. Prepare user_data (see freqtrade repo for required files)
mkdir -p user_data/{strategies,karakana_policy/watch,logs,reports}
cp configbybit.json user_data/
cp Strategy322sgo.py user_data/strategies/
cp best_policy.pt offline_preprocessing.json user_data/karakana_policy/watch/

# 2. Pull image
docker pull moinonin/freqtrade-karakana:1.0.6

# 3. Run with docker-compose-karakana.yml (from freqtrade repo)
docker compose -f docker-compose-karakana.yml up -d

# 4. Verify
docker compose -f docker-compose-karakana.yml ps
# All 4 services: healthy
```

---

## Image Usage Patterns

### 1. Full Stack (4 Services)
Use `docker-compose-karakana.yml` (local) or `docker-compose-cloud.yml` (VPS).
All services share the same image and `user_data` mount.

### 2. One-Off Sweep (No Scheduler)
```bash
docker run --rm \
  -v $(pwd)/user_data:/freqtrade/user_data \
  -v $(pwd)/docker-compose-karakana.yml:/freqtrade/docker-compose-karakana.yml \
  -v /var/run/docker.sock:/var/run/docker.sock \
  moinonin/freqtrade-karakana:1.0.6 \
  python /freqtrade/karakana_tools/sweep_live_shadow_thresholds.py \
  --repo-root /freqtrade \
  --source /freqtrade/user_data/freqtrade_signals_karakana.csv \
  --policy-root /freqtrade/user_data/karakana_policy/watch \
  --long-thresholds 0.02:0.50:0.02 \
  --short-thresholds 0.05:0.60:0.05 \
  --health-window-sizes 128,256,512,1024 \
  --health-window-steps 32,64,128,256 \
  --health-poll-seconds-values 60 \
  --tail-rows 50000 \
  --apply-compose --compose-file /freqtrade/docker-compose-karakana.yml
```

### 3. Shadow Trader Only (Debug)
```bash
docker run --rm \
  -v $(pwd)/user_data:/freqtrade/user_data \
  moinonin/freqtrade-karakana:1.0.6 \
  karakana-shadow-trader \
  --source /freqtrade/user_data/freqtrade_signals_karakana.csv \
  --policy-root /freqtrade/user_data/karakana_policy/watch \
  --output-csv /freqtrade/user_data/reports/karakana_shadow_trade_history.csv \
  --state-path /freqtrade/user_data/reports/karakana_shadow_trader_state.json \
  --horizon-rows 12 --fee-bps 5 \
  --long-threshold 0.02 --short-threshold 0.07 \
  --admission-mode both
```

### 4. Health Monitor Only (Debug)
```bash
docker run --rm \
  -v $(pwd)/user_data:/freqtrade/user_data \
  moinonin/freqtrade-karakana:1.0.6 \
  karakana-shadow-health \
  --source /freqtrade/user_data/reports/karakana_shadow_trade_history.csv \
  --output-csv /freqtrade/user_data/reports/directional_strategy_health.csv \
  --window-size 512 --window-step 256 \
  --base-long-threshold 0.02 --base-short-threshold 0.16
```

---

## Auto-Apply Configuration

The scheduler supports automatic compose patching + sidecar recreation.

**Required mounts for auto-apply:**
```yaml
volumes:
  - ./user_data:/freqtrade/user_data
  - ./docker-compose-karakana.yml:/freqtrade/docker-compose-karakana.yml  # NOT :ro
  - /var/run/docker.sock:/var/run/docker.sock
```

**Env vars:**
```yaml
environment:
  - KARAKANA_THRESHOLD_SCHEDULER_APPLY=1          # Enable compose patching
  - KARAKANA_THRESHOLD_SCHEDULER_RESTART_SIDECARS=1  # Enable docker compose up -d
  # Both must be 1 for full auto-apply
```

**What happens when both = 1:**
1. Sweep runs, finds viable recommendation
2. `_patch_compose()` updates env vars + CLI args in compose file
3. `docker compose -f /freqtrade/docker-compose-karakana.yml up -d --no-deps karakana-shadow-trader karakana-shadow-health`
4. Sidecars recreate with new thresholds

---

## Building The Image

```bash
cd freqtrade-karakana

# Local multi-arch (cached, not pushed)
make build

# Build + push linux/amd64 + linux/arm64, retag :latest
make push TAG=1.0.6

# Arm64-only loaded into local docker
make load-arm64

# Inspect manifest
make inspect TAG=1.0.6
```

**CI:** Push tag `v1.0.6` or run workflow manually. Requires `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets.

---

## Version History

| Tag | Digest | Changes |
|-----|--------|---------|
| 1.0.2 | `sha256:8d6373da...` | First multi-arch |
| 1.0.3 | `sha256:...` | Baked sidecar tools to `/freqtrade/karakana_tools/` |
| 1.0.4 | `sha256:f09bca15...` | Fixed shadow cursor stalling on delisted pairs |
| 1.0.5 | `sha256:ea312059...` | Precomputed pair exit indices (sweep O(n²)→O(n)) |
| 1.0.6 | `sha256:7ee004f6...` | Auto-apply: compose up -d recreation, fixed compose mount |
| 1.0.7 | (pending push) | 2-action `scale_invariant_v1` policy support, 5 bps positive-EV sweep gate (`--min-reward-mean`), skip unchanged sidecar restarts |

---

## Key Implementation Details

### Shadow Trader Cursor Fix (1.0.4)
**Problem:** `last_source_row` landed on delisted pair tail → no horizon-complete exit → stalled forever at `trades=0`.

**Fix:** Collect all evaluable rows per pass, skip stale un-evaluable rows, stop only at live tail row (retries when new candles arrive).

### Sweep Performance Fix (1.0.5)
**Problem:** `_pair_exit_index()` forward scan O(n²) → 50k rows = hours.

**Fix:** `_exit_index_by_row()` builds row→exit map in one grouping pass. 50k sweep now ~2.5 min.

### Auto-Apply Recreation (1.0.6)
**Problem:** `docker restart` doesn't re-read compose file → patched thresholds ignored.

**Fix:** `_restart_sidecars()` uses `docker compose up -d --no-deps` so new env vars / CLI args take effect.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `trades=0` in shadow logs | Cursor stuck on stale pair | Image ≥1.0.4 |
| Sweep hangs >10 min | O(n²) forward scan | Image ≥1.0.5 |
| Compose patched but thresholds unchanged | `docker restart` used | Image ≥1.0.6 + `RESTART_SIDECARS=1` |
| "Read-only file system" on patch | Compose mounted `:ro` | Mount without `:ro` |
| "Docker socket not mounted" | Missing socket mount | Add `-v /var/run/docker.sock:/var/run/docker.sock` |
| Sidecars not recreated | `APPLY=1` but `RESTART_SIDECARS=0` | Set both to 1 |

---

## Support

- **Image issues:** Check `docker logs <container>`
- **Scheduler decisions:** `cat user_data/reports/karakana_threshold_scheduler_state.json`
- **Sweep recommendations:** `cat user_data/reports/karakana_live_threshold_sweep.json`
- **Full ops guide:** See `KARAKANA_OPERATIONS.md` in freqtrade repo