FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV HOME=/app
ENV XDG_CACHE_HOME=/tmp/.cache

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY model_artifacts /app/model_artifacts
COPY entity_type_tagging /app/entity_type_tagging
COPY run.py inference.py /app/

ENTRYPOINT ["python", "run.py"]
