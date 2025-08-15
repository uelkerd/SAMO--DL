FROM python:3.12-alpine3.20

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HF_HOME=/var/tmp/hf-cache \
    XDG_CACHE_HOME=/var/tmp/hf-cache \
    PIP_ROOT_USER_ACTION=ignore

# System deps (ffmpeg and curl) on Alpine
RUN apk add --no-cache ffmpeg=~6.1 curl=~8.7 ca-certificates=~2024 bash=~5.2 shadow=~4.14 \
 && update-ca-certificates

# Install glibc compatibility (needed for manylinux wheels like PyTorch)
ENV GLIBC_VERSION=2.39-r0
RUN wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub \
 && wget -q https://github.com/sgerrand/alpine-pkg-glibc/releases/download/${GLIBC_VERSION}/glibc-${GLIBC_VERSION}.apk \
 && wget -q https://github.com/sgerrand/alpine-pkg-glibc/releases/download/${GLIBC_VERSION}/glibc-bin-${GLIBC_VERSION}.apk \
 && apk add --no-cache glibc-${GLIBC_VERSION}.apk glibc-bin-${GLIBC_VERSION}.apk \
 && rm -f glibc-*.apk

WORKDIR /app

# Python deps
COPY deployment/cloud-run/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip==25.2 \
 && pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ ./src/

# Create and switch to non-root user before healthcheck
RUN adduser -D -s /bin/sh -h /home/app appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Healthcheck (runs as non-root user)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

# Unified API entrypoint
CMD ["sh", "-c", "exec uvicorn src.unified_ai_api:app --host 0.0.0.0 --port ${PORT}"]