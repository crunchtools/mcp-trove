# MCP Trove CrunchTools Container
# Multi-stage build: compile native wheels in builder, copy into Hummingbird
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

# Stage 1: Build wheels (needs gcc for py-rust-stemmers on Python 3.14)
FROM registry.fedoraproject.org/fedora:44 AS builder

RUN dnf install -y python3 python3-pip gcc && dnf clean all

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip wheel --no-cache-dir --wheel-dir=/wheels ".[vision]"

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

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels "mcp-trove-crunchtools[vision]"

RUN python -c "from mcp_trove_crunchtools import main; print('Installation verified')"

ENV TROVE_DB=/tmp/trove-data/trove.db

EXPOSE 8020
ENTRYPOINT ["python", "-m", "mcp_trove_crunchtools"]
