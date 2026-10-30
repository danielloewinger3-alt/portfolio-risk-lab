.PHONY: setup check test lint typecheck fmt app app-smoke data index validate export-reqs

setup:
	uv sync
	./scripts/dev/macos_lightgbm_libomp.sh

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not network" --cov=src/risklab --cov-report=term-missing --cov-report=json:coverage.json
	uv run python scripts/dev/check_coverage.py coverage.json

check: lint typecheck test

app:
	uv run streamlit run app/Home.py

# Headless smoke test: start the app, poll /_stcore/health, then stop it.
app-smoke:
	./scripts/dev/app_smoke.sh

data:
	uv run python scripts/fetch_sample_data.py

index:
	uv run python scripts/build_index.py

validate:
	uv run python scripts/make_validation_report.py

# requirements.txt for pip users; app/requirements.txt (identical) sits next to the entry
# point so Streamlit Community Cloud uses it instead of the root uv.lock.
export-reqs:
	uv export --no-dev --no-hashes --no-emit-project -o requirements.txt
	cp requirements.txt app/requirements.txt
