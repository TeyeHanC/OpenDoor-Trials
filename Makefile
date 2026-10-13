PY := .venv/bin/python

.PHONY: install demo test

# --copies: symlinked venvs do not work on exFAT/FAT volumes.
install:
	test -x $(PY) || python3 -m venv --copies .venv
	$(PY) -m pip install -q -e ".[dev]"

demo: install
	$(PY) -m trialmatch serve

test: install
	$(PY) -m pytest -q
