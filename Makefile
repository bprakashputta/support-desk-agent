.PHONY: setup data test lint fmt run docker eval eval-redteam report

setup:
	uv sync

data:
	uv run python scripts/download_tau_bench.py

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff format . && uv run ruff check . --fix

run:
	uv run uvicorn support_desk.service.app:app --reload --port 8000

docker:
	docker compose up --build

# Full test split, both configurations, then the red-team set and the report (~$1.60 with Haiku 5.5)
eval: data
	uv run python -m support_desk.evals tau --split test --config prompt_only
	uv run python -m support_desk.evals tau --split test --config guarded
	$(MAKE) eval-redteam
	$(MAKE) report

eval-redteam:
	uv run python -m support_desk.evals redteam --config prompt_only
	uv run python -m support_desk.evals redteam --config guarded

report:
	uv run python -m support_desk.evals report
