# Build/publish the Freqtrade + Karakana runtime image.
# Requires: docker with the buildx plugin, and a Docker Hub login with push
# access to moinonin/freqtrade-karakana (docker login).

IMAGE     ?= moinonin/freqtrade-karakana
TAG       ?= 1.0.7
PLATFORMS ?= linux/amd64,linux/arm64
BUILDER   ?= multi-arch-builder

.PHONY: help build push load-arm64 inspect

help:
	@echo "Targets:"
	@echo "  build       - build both platforms into the buildx cache (no push)"
	@echo "  push        - build both platforms and push IMAGE:TAG + IMAGE:latest"
	@echo "  load-arm64  - build linux/arm64 only and load it into the local docker"
	@echo "  inspect     - show the pushed manifest for IMAGE:TAG"
	@echo "Variables: IMAGE=$(IMAGE) TAG=$(TAG) PLATFORMS=$(PLATFORMS) BUILDER=$(BUILDER)"

build:
	docker buildx build \
		--builder $(BUILDER) \
		--platform $(PLATFORMS) \
		-t $(IMAGE):$(TAG) \
		.

push:
	docker buildx build \
		--builder $(BUILDER) \
		--platform $(PLATFORMS) \
		-t $(IMAGE):$(TAG) \
		-t $(IMAGE):latest \
		--provenance=false \
		--push \
		.
	@echo "Pushed $(IMAGE):$(TAG) and $(IMAGE):latest"
	@echo "Get the digest with: make inspect"

load-arm64:
	docker buildx build \
		--builder $(BUILDER) \
		--platform linux/arm64 \
		-t $(IMAGE):$(TAG) \
		--load \
		.

inspect:
	docker buildx imagetools inspect $(IMAGE):$(TAG)
