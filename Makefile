# Usage:
#   make build VERSION=0.0.3rc0  # build wheel + docker image (no tests)
#   make test  VERSION=0.0.3rc0  # smoke-test the built wheel + image

.PHONY: build test validate-version build-wheel smoke-wheel build-docker smoke-docker require-version

VERSION ?=
RELEASE_IMAGE ?= lmsysorg/sglang-jax
WHEEL ?= python/dist/sglang_jax-$(VERSION)-py3-none-any.whl
RELEASE_DOCKER_TAG := $(RELEASE_IMAGE):$(VERSION)

require-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION is required, e.g. make $@ VERSION=0.0.3rc0"; \
		exit 1; \
	fi

validate-version: require-version
	@python -c "from packaging.version import Version, InvalidVersion; \
import sys; \
v = Version('$(VERSION)'); \
print(f'parsed: {v} (is_prerelease={v.is_prerelease})')"

build-wheel: validate-version
	cd python && cp -f ../README.md ../LICENSE .
	python -m pip install --upgrade pip build
	cd python && SETUPTOOLS_SCM_PRETEND_VERSION=$(VERSION) python -m build
	@test -f $(WHEEL) || (echo "Error: expected $(WHEEL) was not produced"; exit 1)
	@echo "Built $(WHEEL)"

build-docker: validate-version
	docker buildx build \
		--platform linux/amd64 \
		--build-arg SETUPTOOLS_SCM_PRETEND_VERSION=$(VERSION) \
		--load \
		-t $(RELEASE_DOCKER_TAG) \
		-f Dockerfile .

smoke-wheel: validate-version
	python -m pip install --force-reinstall $(WHEEL)
	python -c "import sgl_jax; \
print('sgl_jax version:', sgl_jax.__version__); \
assert sgl_jax.__version__ == '$(VERSION)', sgl_jax.__version__"

smoke-docker: validate-version
	docker run --rm $(RELEASE_DOCKER_TAG) python -c "import sgl_jax; \
print('sgl_jax version:', sgl_jax.__version__); \
assert sgl_jax.__version__ == '$(VERSION)', sgl_jax.__version__"

build: build-wheel build-docker
	@echo "Built wheel and docker image for $(VERSION)"

test: smoke-wheel smoke-docker
	@echo "Smoke tests passed for $(VERSION)"
