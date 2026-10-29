# syntax=docker/dockerfile:1
# Reuse the tested CUDA 13 / PyTorch 2.13 runtime and Runpod SSH initialization.
# This base contains application code and dependencies, not model weights.
FROM ghcr.io/akhilsivanand/moss-comfyui@sha256:21cde4857330e4bd5f8660dc95a89a85c358c79fbac77e2c08f0e80004729671
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG COMFY_COMMIT=426f5eea52a7ece1dd599fff4cb6cd1f352ddd24
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="Fibo 2 Edit Turbo ComfyUI" \
      org.opencontainers.image.description="Native Fibo 2 image editing for ComfyUI and Runpod" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.licenses="MIT"

WORKDIR /opt
# These are baked application paths in this new image, never a user volume.
RUN rm -rf /opt/moss /opt/comfyui
COPY requirements/ /opt/fibo/requirements/
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-compile \
      --only-binary=:all: --require-hashes --no-deps -r /opt/fibo/requirements/upgrade.lock \
    && /opt/venv/bin/python -m pip check
RUN git init --quiet /opt/comfyui \
    && git -C /opt/comfyui remote add origin https://github.com/Comfy-Org/ComfyUI.git \
    && git -C /opt/comfyui fetch --depth 1 origin "$COMFY_COMMIT" \
    && git -C /opt/comfyui checkout --detach "$COMFY_COMMIT" \
    && test "$(git -C /opt/comfyui rev-parse HEAD)" = "$COMFY_COMMIT" \
    && printf '%s\n' "$COMFY_COMMIT" >/opt/fibo/comfy-commit.txt \
    && rm -rf /opt/comfyui/.git

COPY src/ /opt/fibo/src/
COPY config/ /opt/fibo/config/
COPY workflows/ /opt/fibo/workflows/
COPY scripts/ /opt/fibo/scripts/
COPY handler.py README.md LICENSE THIRD_PARTY_NOTICES.txt /opt/fibo/
COPY custom_nodes/fibo/ /opt/comfyui/custom_nodes/fibo/
COPY scripts/base-ready.sh /post_start.sh
RUN ln -s /opt/fibo/workflows/ui /opt/comfyui/custom_nodes/fibo/example_workflows \
    && chmod 0755 /post_start.sh /opt/fibo/scripts/entrypoint.sh \
    && test -x /start.sh \
    && /opt/venv/bin/python /opt/fibo/scripts/check_native_dependencies.py --root /opt/venv --cuda-only

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/fibo/src \
    PYTHONUNBUFFERED=1 \
    FIBO_ROOT=/opt/fibo \
    COMFY_ROOT=/opt/comfyui \
    FIBO_DATA_DIR=/workspace/ComfyUI \
    FIBO_MODE=pod \
    COMFY_URL=http://127.0.0.1:8189
WORKDIR /opt/fibo
EXPOSE 8888 22
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD ["/opt/venv/bin/python", "/opt/fibo/scripts/healthcheck.py"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/fibo/scripts/entrypoint.sh"]
