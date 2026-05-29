IMAGE_NAME ?= sglang-jax
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
USERNAME := $(shell whoami)
IMAGE_TAG := $(TIMESTAMP)-$(USERNAME)

REGISTRY := asia-northeast1-docker.pkg.dev/tpu-service-473302/sglang-project
REMOTE_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: build push

build:
	docker build --platform linux/amd64 -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "Built image: $(IMAGE_NAME):$(IMAGE_TAG)"

push: build
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REMOTE_IMAGE)
	docker push $(REMOTE_IMAGE)
	@echo "Pushed image: $(REMOTE_IMAGE)"
