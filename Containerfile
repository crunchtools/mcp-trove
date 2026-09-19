# MCP Trove CrunchTools Container
# Multi-stage build: everything that executes runs in the builder; the
# runtime stage is distroless and only COPYs.
#
# Build:
#   podman build -t quay.io/crunchtools/mcp-trove .
#
# Run:
#   podman run -v trove-data:/data -v ~/Documents:/docs:ro quay.io/crunchtools/mcp-trove
#
# With Claude Code:
#   claude mcp add mcp-trove-crunchtools \
#     -- podman run -i --rm -v trove-data:/data -v ~/Documents:/docs:ro quay.io/crunchtools/mcp-trove

# Stage 1: Builder (has a shell, dnf and build tools)
FROM quay.io/hummingbird/python:latest-builder AS builder

USER root
RUN dnf install -y gcc && dnf clean all

WORKDIR /build
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[vision]"

# Bake the embedding model in, so there is no HuggingFace fetch at runtime.
# Cached at an explicit path rather than fastembed's default of
# $TMPDIR/fastembed_cache: /tmp is the one directory a deployment is likely
# to mount over, and that would silently send the model back to the network.
ENV FASTEMBED_CACHE_PATH=/fastembed-cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
    && echo "Embedding model cached"

# Stage 2: Runtime image (minimal, no build tools)
FROM quay.io/hummingbird/python:latest

LABEL name="mcp-trove-crunchtools" \
      version="0.2.0" \
      summary="Self-hosted local file indexing MCP server with semantic search" \
      description="Index local directories and search over contents using hybrid vector + keyword search" \
      maintainer="crunchtools.com" \
      url="https://github.com/crunchtools/mcp-trove" \
      io.k8s.display-name="MCP Trove CrunchTools" \
      io.openshift.tags="mcp,rag,semantic-search,embeddings" \
      org.opencontainers.image.source="https://github.com/crunchtools/mcp-trove" \
      org.opencontainers.image.description="Self-hosted local file indexing MCP server with semantic search" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

WORKDIR /app

# libstdc++ from Hummingbird builder (not Fedora — different glibc)
COPY --from=quay.io/hummingbird/python:latest-builder /usr/lib64/libstdc++.so.6* /usr/lib64/
COPY --from=builder /app/venv /app/venv
COPY --from=builder --chown=65532:65532 /fastembed-cache /fastembed-cache
ENV PATH="/app/venv/bin:$PATH"
ENV FASTEMBED_CACHE_PATH=/fastembed-cache

# Verify the install. Exec form: this stage has no /bin/sh for RUN's shell form.
RUN ["python3", "-c", "from mcp_trove_crunchtools import main; print('Installation verified')"]

ENV TROVE_DB=/tmp/trove-data/trove.db
ENV OMP_NUM_THREADS=4

EXPOSE 8020
ENTRYPOINT ["python", "-m", "mcp_trove_crunchtools"]
