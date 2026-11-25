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
FROM quay.io/hummingbird/python:3.14-builder AS builder

USER root
RUN dnf install -y gcc python3-devel make pkgconf diffutils gawk tar gzip xz libdav1d-devel && dnf clean all

# FFmpeg, built here rather than taken from PyAV's wheel. The wheel bundles
# the FFmpeg that PyAV-Org/pyav-ffmpeg last built, which lags FFmpeg's
# security releases (19.0.2 carries 9.0.2, with criticals fixed in 9.1), and
# Hummingbird does not package FFmpeg or LAME. Both tarballs are pinned by
# hash; FFmpeg's was checked against its release signing key (B4322F04D67658D8).
# No nasm in Hummingbird, so no x86 asm: sampling ten frames and twenty
# seconds of sound does not need it. libmp3lame is the clip's MP3 encoder,
# libdav1d its AV1 decoder. The rpath lets the runtime stage find all of it
# under /opt/ffmpeg without an LD_LIBRARY_PATH.
ARG LAME_VERSION=3.100
ARG LAME_SHA256=ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e
ARG FFMPEG_VERSION=9.1
ARG FFMPEG_SHA256=1a4d394ae63d190a526cd052eb4415c156b5c5e2df37efab41a7de0b8fde2433
WORKDIR /src
RUN curl -fsSL -o lame.tar.gz "https://downloads.sourceforge.net/project/lame/lame/${LAME_VERSION}/lame-${LAME_VERSION}.tar.gz" \
    && echo "${LAME_SHA256}  lame.tar.gz" | sha256sum -c - \
    && tar xzf lame.tar.gz && cd "lame-${LAME_VERSION}" \
    && sed -i "/lame_init_old/d" include/libmp3lame.sym \
    && ./configure --prefix=/opt/ffmpeg --enable-shared --disable-static --disable-frontend --disable-decoder \
    && make -j"$(nproc)" && make install \
    && cd /src && curl -fsSL -o ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
    && echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c - \
    && tar xJf ffmpeg.tar.xz && cd "ffmpeg-${FFMPEG_VERSION}" \
    && ./configure --prefix=/opt/ffmpeg --enable-shared --disable-static \
         --disable-programs --disable-doc --disable-x86asm --disable-autodetect \
         --enable-libmp3lame --enable-libdav1d \
         --extra-cflags=-I/opt/ffmpeg/include --extra-ldflags="-L/opt/ffmpeg/lib -Wl,-rpath,/opt/ffmpeg/lib" \
    && make -j"$(nproc)" && make install \
    && cp -a /usr/lib64/libdav1d.so.* /opt/ffmpeg/lib/ \
    && rm -rf /src /opt/ffmpeg/share
ENV PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig \
    LDFLAGS=-Wl,-rpath,/opt/ffmpeg/lib

WORKDIR /build
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
COPY pyproject.toml README.md uv.lock ./
COPY src/ ./src/

# Install exactly what uv.lock pins, hash-checked, then the project itself
# without re-resolving. Building from the open ranges in pyproject.toml let
# an upstream release change the image underneath us (pymupdf4llm 2.0,
# RT #1542); dependency bumps now arrive as lockfile PRs through CI.
# PyAV alone builds from its sdist, still hash-checked, against the FFmpeg
# above.
RUN pip install --no-cache-dir uv==0.12.23 \
    && uv export --frozen --no-dev --extra vision --no-emit-project -o requirements.txt \
    && pip install --no-cache-dir --require-hashes --no-deps --no-binary av -r requirements.txt \
    && pip install --no-cache-dir --no-deps . \
    && pip uninstall -y uv

# Bake the embedding model in, so there is no HuggingFace fetch at runtime.
# Cached at an explicit path rather than fastembed's default of
# $TMPDIR/fastembed_cache: /tmp is the one directory a deployment is likely
# to mount over, and that would silently send the model back to the network.
ENV FASTEMBED_CACHE_PATH=/fastembed-cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
    && echo "Embedding model cached"

# Stage 2: Runtime image (minimal, no build tools)
FROM quay.io/hummingbird/python:3.14

LABEL name="mcp-trove-crunchtools" \
      version="1.2.4" \
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

COPY --from=builder /opt/ffmpeg /opt/ffmpeg
COPY --from=builder /app/venv /app/venv
COPY --from=builder --chown=65532:65532 /fastembed-cache /fastembed-cache
ENV PATH="/app/venv/bin:$PATH"
ENV FASTEMBED_CACHE_PATH=/fastembed-cache

# Verify the install. Exec form: this stage has no /bin/sh for RUN's shell form.
# Import the PDF stack by name: extractor.py imports it lazily, so a hollow
# pymupdf4llm (2.0 shipped as an empty shim, RT #1542) would otherwise pass
# this check and fail every PDF at runtime.
RUN ["python3", "-c", "from mcp_trove_crunchtools import main; import pymupdf, pymupdf4llm; pymupdf4llm.to_markdown; import docx, pptx, openpyxl, xlrd, olefile, striprtf.striprtf, defusedxml, av; av.Codec('mp3', 'w'); assert av.ffmpeg_version_info == '9.1', av.ffmpeg_version_info; print('Installation verified')"]

ENV TROVE_DB=/tmp/trove-data/trove.db

EXPOSE 8020
ENTRYPOINT ["python", "-m", "mcp_trove_crunchtools"]
