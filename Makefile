.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

clean-pyc: ## Remove python artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
.PHONY: clean-pyc

venv: ## create virtual environment
	python3.8 -m venv venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/python -m pip install --upgrade setuptools
	venv/bin/python -m pip install --upgrade wheel
.PHONY: venv

dependencies:  ## Install dependencies from requirements.txt
	pip install -r requirements.txt
.PHONY: dependencies

test-dependencies: ## Install dependencies from test_requirements.txt
	pip install -r test_requirements.txt
.PHONY: test-dependencies

update-dependencies:  ## Update dependency versions
	pip install pip-tools
	pip-compile requirements.in > requirements.txt
	pip-compile test_requirements.in > test_requirements.txt
	pip-compile service_requirements.in > service_requirements.txt
.PHONY: update-dependencies

clean-venv: ## remove all packages from virtual environment
	pip freeze | grep -v "^-e" | xargs pip uninstall -y
.PHONY: clean-venv

clean-test:	## Remove test artifacts
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf reports
	rm -rf .pytype
.PHONY: clean-test

check-codestyle:  ## checks the style of the code against PEP8
	pycodestyle policy_decorator --max-line-length=120
.PHONY: check-codestyle

check-docstyle:  ## checks the style of the docstrings against PEP257
	pydocstyle policy_decorator
.PHONY: check-docstyle

check-security:  ## checks for common security vulnerabilities
	bandit -r policy_decorator
.PHONY: check-security

check-dependencies:  ## checks for security vulnerabilities in dependencies
	safety check -r requirements.txt
.PHONY: check-dependencies

check-codemetrics:  ## calculate code metrics of the package
	radon cc policy_decorator
.PHONY: check-codemetrics

check-pytype:  ## perform static code analysis
	pytype policy_decorator
.PHONY: check-pytype

convert-post:  ## Convert the notebook into Markdown file
	jupyter nbconvert --to markdown blog_post/blog_post.ipynb --output-dir='./blog_post' --TagRemovePreprocessor.remove_input_tags='{"hide_code"}'
.PHONY: convert-post

build-image:  ## Build docker image
	export BUILD_DATE=`date -u +'%Y-%m-%dT%H:%M:%SZ'` && \
	docker build --build-arg BUILD_DATE \
		-t diabetes_risk_model_service:latest .
.PHONY: build-image
