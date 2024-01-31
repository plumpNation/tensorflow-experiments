install:
	pip install -r ./requirements.txt

venv-create:
	python -m venv .

venv-use:
	source bin/activate

freeze-deps:
	pip freeze > requirements.txt
