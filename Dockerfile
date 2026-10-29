# syntax=docker/dockerfile:1
# Reuse the exact GPU-tested Fibo inference environment. Only application
# workflows and standard Runpod service wiring change in this derivative.
FROM ghcr.io/akhilsivanand/fibo-comfyui@sha256:7487d8ab9784c0f8b4176b21332e0a3f6c762053b597379d3a94e44dfcae46d2
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG COMFY_COMMIT=426f5eea52a7ece1dd599fff4cb6cd1f352ddd24
ARG VERSION=0.2.0
LABEL org.opencontainers.image.title="Fibo 2 Edit Turbo ComfyUI" \
      org.opencontainers.image.description="Native Fibo 2 workflows with standard Runpod service ports" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.licenses="MIT"

WORKDIR /opt
COPY requirements/ /opt/fibo/requirements/
RUN mkdir -p /workspace \
    && test "$(cat /opt/fibo/comfy-commit.txt)" = "$COMFY_COMMIT" \
    && /opt/venv/bin/python -m pip check

COPY src/ /opt/fibo/src/
COPY config/ /opt/fibo/config/
COPY workflows/ /opt/fibo/workflows/
COPY scripts/ /opt/fibo/scripts/
COPY handler.py README.md LICENSE THIRD_PARTY_NOTICES.txt /opt/fibo/
COPY custom_nodes/fibo/ /opt/comfyui/custom_nodes/fibo/
COPY scripts/base-ready.sh /post_start.sh
RUN bash /opt/fibo/scripts/install_jupyter.sh
RUN bash /opt/fibo/scripts/install_filebrowser.sh
RUN test "$(readlink /opt/comfyui/custom_nodes/fibo/example_workflows)" = /opt/fibo/workflows/ui \
    && chmod 0755 /post_start.sh /opt/fibo/scripts/entrypoint.sh \
    && test -x /start.sh

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/opt/venv/bin:/opt/jupyter/bin:$PATH \
    PYTHONPATH=/opt/fibo/src \
    PYTHONUNBUFFERED=1 \
    FIBO_ROOT=/opt/fibo \
    COMFY_ROOT=/opt/comfyui \
    FIBO_DATA_DIR=/workspace/ComfyUI \
    FIBO_MODE=pod \
    COMFY_URL=http://127.0.0.1:8189
WORKDIR /opt/fibo
EXPOSE 8188 8080 8888 22
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD ["/opt/venv/bin/python", "/opt/fibo/scripts/healthcheck.py"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/fibo/scripts/entrypoint.sh"]
