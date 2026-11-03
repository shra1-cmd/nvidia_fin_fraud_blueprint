# Fraud-scoring API image. The CUDA runtime comes inside the torch / xgboost wheels, so the base
# image needs no CUDA: the host only needs an NVIDIA driver + nvidia-container-toolkit.
#
#   TORCH_INDEX  cu126 (default): driver >= 525.60.13, supports V100 (sm_70)
#                cu118          : older drivers (>= 450.80.02); see CLAUDE.md step 2
#   Do NOT use cu128 / cu13x on a V100: those wheels have no sm_70 kernels.
#
# Artifacts are NOT baked in: docker-compose mounts ./artifacts read-only at /app/artifacts.
FROM python:3.12.7-slim-bookworm

ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX=cu126

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN pip install --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" "torch==${TORCH_VERSION}"
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/

RUN useradd --create-home --uid 10001 fraud
USER fraud

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=4)"

# One worker: each worker would load its own copy of the models and graph onto the GPU.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
