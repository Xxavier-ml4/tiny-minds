.PHONY: test test-verbose lint cli-help clean

# Runs the full suite with the standard library's unittest runner, so it
# works with zero installed dependencies beyond pyyaml. If pytest is
# installed (`pip install -e .[dev]`), `pytest` also collects and runs the
# same test files.
test:
	python3 -m unittest discover -s tests -p 'test_*.py' -v

test-verbose:
	python3 -m unittest discover -s tests -p 'test_*.py' -v 2>&1 | tail -100

lint:
	python3 -m py_compile $$(find tinymind -name '*.py')
	@echo "py_compile OK (install ruff/mypy for real linting: pip install -e .[dev])"

cli-help:
	python3 -m tinymind.cli --help

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info .pytest_cache
