.PHONY:	install lint fix test check sweep

PYTEST_ARGS ?= -q

install:
	pip install -e ."[datasets,dev]"

lint:
	ruff check src tests scripts

fix: 
	ruff check src tests scripts --fix 

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest $(PYTEST_ARGS)

check:	fix test

sweep:
	python3 scripts/run_transfer_sweep.py





