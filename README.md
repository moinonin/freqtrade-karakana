# freqtrade-karakana

Standalone build repo for the **Freqtrade + Karakana** runtime Docker image,
published as [`moinonin/freqtrade-karakana`](https://hub.docker.com/r/moinonin/freqtrade-karakana).

This repo is intentionally tiny: it contains only what the image needs.
Freqtrade itself is pulled from [`freqtradeorg/freqtrade:2026.3`](https://hub.docker.com/r/freqtradeorg/freqtrade);
this build layers the Karakana Python package and the two sidecar CLIs on top.

## What's in here

| Path | Role |
|---|---|
| `Dockerfile` | Two-stage build: install the Karakana wheel into the Freqtrade user env, then copy it into a clean Freqtrade base |
| `pyproject.toml` | Karakana wheel metadata (console scripts: `karakana-shadow-trader`, `karakana-shadow-health`) |
| `karakana/` | The Python package (SGO math, runtime sidecars) |
| `config/karakana-config.json` | Copied to `/freqtrade/karakana/` in the image |
| `Makefile` | `make build` / `make push` / `make load-arm64` |
| `.github/workflows/docker.yml` | CI: builds and pushes `linux/amd64` + `linux/arm64` on `v*` tags or manual dispatch |

The image pins these runtime deps regardless of the Freqtrade base:

- `torch==2.11.0+cpu` (PyTorch CPU wheel index)
- `numpy==2.2.6`, `pandas==2.3.3`
- `ccxt==4.5.54` (with `setuptools<82` to satisfy torch's constraint)
- `gymnasium==1.2.3`
- `ft-scikit-optimize==0.9.2`, `termcolor`, `pygments`

## Build & publish

```sh
# Local multi-arch build (cached, not pushed)
make build

# Build + push linux/amd64 and linux/arm64, retag :latest
make push TAG=1.0.3

# Arm64-only image loaded into the local docker for smoke tests
make load-arm64

# Show the pushed manifest and per-platform digests
make inspect TAG=1.0.3
```

CI publishing: push a tag `v1.0.3` (or run the workflow manually with a tag).
Requires repo secrets `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` with write
access to the Docker Hub repo.

## Deploying

Deployments are driven from the Freqtrade repo:

- `scripts/deploy_cloud.sh <user@vps> <config.json> [image]` syncs the stack to
  a VPS and pins the pulled `@sha256:` digest in the remote `.env`.
- `docker-compose-cloud.yml` runs `freqtrade trade` plus the two Karakana
  sidecars (`karakana-shadow-trader`, `karakana-shadow-health`).

Strategy files, SQLite DBs, exchange configs, and Karakana policies stay out
of this image — they live in the mounted `user_data/` volume.

## Known quirk

`karakana/config.py` resolves `config/karakana-config.json` relative to the
CWD (`/freqtrade/config/...`) or the package parent — neither exists in the
image, so `load_config()` silently falls back to its built-in defaults. The
config copied to `/freqtrade/karakana/` is therefore inert unless something
sets `KARAKANA_CONFIG=/freqtrade/karakana/karakana-config.json`. Keep this in
mind when tuning: profile overrides belong in the mounted `user_data` config
or via the env var, not in this image.
