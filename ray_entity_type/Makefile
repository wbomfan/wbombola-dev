DOCKER ?= docker
REGION = us-east-1
REGISTRY = 170047911957.dkr.ecr.us-east-1.amazonaws.com
IMAGE = mle-entity-type-ray-dev
TAG ?= latest
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

MODEL_S3_PATH ?= s3://mle-mlflow-artifacts-wikia-v3/4/81a9a9f6d8f146e7ae2de5bb474b2be5/artifacts/model/
INPUT_S3_PATH ?= s3://de-analytics-temp/wbombola/entity_type_ray/datasets/input_increment/data/
OUTPUT_S3_PATH ?= s3://de-analytics-temp/wbombola/entity_type_ray/output/$(TIMESTAMP)/
BATCH_SIZE ?= 128
NUM_ACTORS ?= 1
LIMIT ?= 50

RUN_ARGS = --input_s3_path $(INPUT_S3_PATH) --output_s3_path $(OUTPUT_S3_PATH) \
	--batch_size $(BATCH_SIZE) --num_actors $(NUM_ACTORS) $(if $(LIMIT),--limit $(LIMIT))

download_model:
	aws s3 cp --recursive $(MODEL_S3_PATH) model_artifacts/

auth:
	aws ecr get-login-password --region $(REGION) | $(DOCKER) login --username AWS --password-stdin $(REGISTRY)

build:
	DOCKER_DEFAULT_PLATFORM=linux/amd64 $(DOCKER) build -t $(REGISTRY)/$(IMAGE):$(TAG) .

push: auth
	$(DOCKER) push $(REGISTRY)/$(IMAGE):$(TAG)

pull: auth
	$(DOCKER) pull $(REGISTRY)/$(IMAGE):$(TAG)

run_local:
	$(DOCKER) run --rm --gpus all --shm-size=10g --network host $(REGISTRY)/$(IMAGE):$(TAG) $(RUN_ARGS)

run_cpu:
	python run.py $(RUN_ARGS) --num_gpus 0

submit_batch_dev:
	python submit_batch_job.py --config config_dev.yaml --env_var_prefix ETR \
		--batch_job_name_prefix entity-type-ray-dev --observe

.PHONY: download_model auth build push pull run_local run_cpu submit_batch_dev
