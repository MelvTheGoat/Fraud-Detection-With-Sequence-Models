# Public demo image: the scoring API plus a browser front end.
#
# This is NOT the production image. docker-compose.yml and the root Dockerfile
# mount the model bundle read-only at run time, so the image stays independent of
# any single training run. A Hugging Face Space has no volumes, so this image
# bakes the bundle in and is therefore tied to one run — which is correct for a
# demo and wrong for production. Keeping the two Dockerfiles separate is cheaper
# than one file with a branch in it.
#
# Build context is the repository root:
#     docker build -f deploy/Dockerfile -t fraud-demo .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# torch is needed even though scoring runs on ONNX Runtime: the explanation path
# reads attention or gradients from the eager model, and /health's fallback runs
# the PyTorch module when the ONNX session will not start. The CPU-only wheel is
# about a quarter the size of the default one, which pulls a CUDA runtime onto a
# machine that has no GPU.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<3"

COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY pyproject.toml ./
COPY src ./src
COPY deploy ./deploy

# The bundle: feature builder, checkpoint, ONNX graph, calibrator, descriptor.
# Produced by `make experiment`; see deploy/README.md.
COPY artifacts ./artifacts

ENV FRAUD_ARTIFACT_DIR=/app/artifacts \
    FRAUD_AUDIT_PATH=/app/audit/decisions.jsonl \
    FRAUD_DEMO_DATA=/app/deploy/demo_data.json

# Hugging Face Spaces runs the container as uid 1000 and mounts nothing, so the
# audit directory has to exist and be writable at build time. On a Space the
# audit log is ephemeral — it dies with the container — which is fine for a demo
# and would be a compliance failure in production. Said out loud in the README.
RUN useradd --create-home --uid 1000 scoring && \
    mkdir -p /app/audit && \
    chown -R scoring:scoring /app
USER scoring

# 7860 is the port Spaces expects by default; it is also declared in the Space
# README's app_port so this stays explicit rather than conventional.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:7860/health')); \
sys.exit(0 if r['model_loaded'] else 1)"

CMD ["uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
