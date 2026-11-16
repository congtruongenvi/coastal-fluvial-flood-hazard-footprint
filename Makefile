.PHONY: help venv test clean

help:
	@echo "make venv   - create the environment and install requirements"
	@echo "make test   - run the self-tests (no data needed)"
	@echo "make clean  - remove generated outputs"

venv:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/Scripts/python.exe -r requirements.txt

test:
	.venv/Scripts/python.exe src/selftest.py

clean:
	rm -f figures/*.png data/processed/*
