FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY prompts ./prompts
COPY kb ./kb
COPY scripts ./scripts
RUN uv sync --locked --no-dev

RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app/data
USER app
ENV DATA_DIR=/app/data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
# Seed data (tau-bench retail, ~2 MB) is fetched on first start into the data volume.
CMD ["sh", "-c", "[ -f /app/data/tau_bench/orders.json ] || uv run --no-sync python scripts/download_tau_bench.py --out /app/data/tau_bench; exec uv run --no-sync uvicorn support_desk.service.app:app --host 0.0.0.0 --port 8000"]
