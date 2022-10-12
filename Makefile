#* Installation
.PHONY: install
install:
	python setup.py install

.PHONY: develop
develop:
	python setup.py develop

.PHONY: pre
pre:
	pre-commit run -a

.PHONY: pre-install
pre-install:
	pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	isort --settings-path pyproject.toml ./
	black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	pytest -c pyproject.toml --cov-report=html --cov=imzy tests/
	coverage-badge -o assets/images/coverage.svg -f

.PHONY: check-codestyle
check-codestyle:
	isort --diff --check-only --settings-path pyproject.toml ./
	black --diff --check --config pyproject.toml ./
	darglint --verbosity 2 imzy tests

.PHONY: mypy
mypy:
	mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	safety check --full-report
	bandit -ll --recursive imzy tests

.PHONY: lint
lint: test check-codestyle mypy check-safety

.PHONY: untrack
untrack:
	git rm -r --cached .
	git add .
	git commit -m ".gitignore fix"
