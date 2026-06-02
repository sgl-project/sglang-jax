IMAGE_NAME ?= sglang-jax
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
USERNAME := $(shell whoami)
IMAGE_TAG := $(TIMESTAMP)-$(USERNAME)

REGISTRY := asia-northeast1-docker.pkg.dev/tpu-service-473302/sglang-project
REMOTE_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: build push release validate-version build-wheel smoke-wheel build-docker smoke-docker require-version

build:
	docker build --platform linux/amd64 -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "Built image: $(IMAGE_NAME):$(IMAGE_TAG)"

push: build
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REMOTE_IMAGE)
	docker push $(REMOTE_IMAGE)
	@echo "Pushed image: $(REMOTE_IMAGE)"

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
	@python -c "from packaging.version import Version; \
v = Version('$(VERSION)'); \
print(f'parsed: {v} (is_prerelease={v.is_prerelease})')"

build-wheel: validate-version
	cd python && cp -f ../README.md ../LICENSE .
	python -m pip install --upgrade pip build
	cd python && SETUPTOOLS_SCM_PRETEND_VERSION=$(VERSION) python -m build
	@test -f $(WHEEL) || (echo "Error: expected $(WHEEL) was not produced"; exit 1)
	@echo "Built $(WHEEL)"

smoke-wheel: build-wheel
	python -m pip install --force-reinstall $(WHEEL)
	python -c "import sgl_jax; \
print('sgl_jax version:', sgl_jax.__version__); \
assert sgl_jax.__version__ == '$(VERSION)', sgl_jax.__version__"

build-docker: validate-version
	docker buildx build \
		--platform linux/amd64 \
		--build-arg SETUPTOOLS_SCM_PRETEND_VERSION=$(VERSION) \
		--load \
		-t $(RELEASE_DOCKER_TAG) \
		-f Dockerfile .

smoke-docker: build-docker
	docker run --rm $(RELEASE_DOCKER_TAG) python -c "import sgl_jax; \
print('sgl_jax version:', sgl_jax.__version__); \
assert sgl_jax.__version__ == '$(VERSION)', sgl_jax.__version__"

release: smoke-wheel smoke-docker
	@echo "Release validation passed for $(VERSION)"
