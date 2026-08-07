# Serving image for the fraud scoring API.
#
# CPU only, deliberately. The deployed model is ~100k parameters over a
# 32-position window at batch size 1; a GPU would add PCIe latency and cost to
# something that already meets its budget on a CPU core.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

# The CPU-only torch wheel is a fraction of the size of the default one, which
# pulls in the CUDA runtime for a machine that has no GPU.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"

COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY pyproject.toml ./
COPY src ./src

# Artefacts (weights, ONNX graph, feature builder, deployment descriptor) are
# mounted rather than baked in, so the image is independent of any single
# training run and a model can be rolled forward without a rebuild.
ENV FRAUD_ARTIFACT_DIR=/app/artifacts \
    FRAUD_AUDIT_PATH=/app/audit/decisions.jsonl

RUN useradd --create-home --uid 10001 scoring && \
    mkdir -p /app/artifacts /app/audit && \
    chown -R scoring:scoring /app
USER scoring

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if r['model_loaded'] else 1)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
