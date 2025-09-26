PYTHON ?= python3
VENV := .venv
VENV_BIN := $(VENV)/bin
VENV_MARKER := $(VENV_BIN)/activate
INSTALL_MARKER := $(VENV)/.installed

.PHONY: install run test clean distclean package

$(VENV_MARKER):
	$(PYTHON) -m venv $(VENV)

$(INSTALL_MARKER): $(VENV_MARKER) pyproject.toml
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e .
	touch $(INSTALL_MARKER)

install: $(INSTALL_MARKER)

run: $(INSTALL_MARKER)
	$(VENV_BIN)/terminal-renderer --fps 24 --fov 70 --scale 2.0

test: $(INSTALL_MARKER)
	$(VENV_BIN)/python -m unittest discover -s tests

package: $(INSTALL_MARKER)
	$(VENV_BIN)/pip install --upgrade build
	$(VENV_BIN)/python -m build

clean:
	rm -rf build dist .pytest_cache *.egg-info
	rm -f $(INSTALL_MARKER)
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV)
