install:
	pip install -e .[datasets,dev]

test:
	pytest -q

lint:
	ruff check src tests
