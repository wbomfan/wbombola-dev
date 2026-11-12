# entity-type-ray

Ray Data pipeline that runs the prod `entityType` BERT model (MLflow `entityType` v2) over an
article parquet dataset on GPU and writes `wiki_id, article_id, tag_key, tag_value, confidence_score`
as parquet.

The code does not know where it runs. `ray.init()` picks up every GPU on the machine, so the same
image runs on a 1-GPU box, a 4-GPU box (set `NUM_ACTORS=4`) or an AWS Batch job.

## Stack

| | |
|---|---|
| base image | `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime` (Python 3.11) |
| ray | 2.53.0 |
| transformers | 4.26.0 (hard pin: the pickled model does not load on 4.49) |
| model | `s3://mle-mlflow-artifacts-wikia/26/e0a17346ba664dd9a16ae61b0569d645/artifacts/model/` baked into the image |

## Files

- `run.py` reads parquet, drops empty rows, runs the GPU actor pool, writes parquet. Options via CLI or `ETR_*` env vars.
- `inference.py` loads the model once per actor and predicts a pandas batch.
- `entity_type_tagging/` model class module the pickle refers to. Do not rename.
- `Makefile` build, push, run local, submit to Batch.
- `config_dev.yaml` + `submit_batch_job.py` Batch submission, same mechanism as genre.

## Local run on a GPU EC2 (manual)

```bash
make download_model          # once, ~400 MB into model_artifacts/
make build push              # from your laptop; DOCKER=podman make build if needed
# on the instance:
make pull
make run_local LIMIT=1000    # smoke run
make run_local NUM_ACTORS=4  # multi-GPU box
```

Ray dashboard is on `http://<instance>:8265` while a run is active (container uses host network).

CPU run without Docker (slow, for debugging):

```bash
pip install -r requirements.txt
make run_cpu LIMIT=40 INPUT_S3_PATH=some/local.parquet OUTPUT_S3_PATH=/tmp/out/
```

## AWS Batch

Terraform, module `mle_batch_worker`:

```hcl
module "entity_type_ray_dev" {
  source              = "../modules/mle_batch_worker"
  worker_name         = "entity-type-ray"
  environment         = "dev"
  image_tag           = "latest"
  platform            = "EC2"
  set_gpu             = true
  vcpu                = 8
  memory              = 30000
  shared_memory_size  = 10240
  job_timeout_seconds = 43200
  s3_buckets_readwrite = ["de-analytics-temp"]
}
```

That creates ECR `entity-type-ray-dev` and job definition `entity-type-ray-job-def-dev`, which
`config_dev.yaml` points at. Queue is the shared `mle-gpu-inference-job-queue`.

```bash
make submit_batch_dev
```

Edit `output_s3_path` in `config_dev.yaml` between runs. Ray writes new files next to old ones.
