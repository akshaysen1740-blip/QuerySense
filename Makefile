.PHONY: install install-ml data api ui dev test lint format typecheck eval check docker

install:      ## Core install with dev tools
	pip install -e ".[dev]"

install-ml:   ## Pretrained bi-encoder, cross-encoder, BERTopic, FAISS
	pip install -e ".[dev,ml]"

data:         ## Regenerate data/sample_tickets.csv and data/golden_queries.json
	python scripts/generate_sample_data.py

api:          ## FastAPI on :8000 (docs at /docs)
	uvicorn signalsift.api.main:app --reload --port 8000

ui:           ## Streamlit on :8501 (in-process mode)
	streamlit run app.py

dev:          ## UI talking to a running API
	SIGNALSIFT_UI_MODE=api streamlit run app.py

test:
	pytest --cov=signalsift --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

typecheck:
	mypy

eval:         ## Evaluate all retrievers; writes reports/
	python scripts/run_eval.py

check: lint typecheck test

docker:
	docker compose up --build
