install:
	@pip install -r ./requirements.txt

venv-create:
	@python -m venv .

venv-use:
	@echo "You need to run:"
	@echo "source bin/activate"

freeze-deps:
	@pip freeze > requirements.txt

1-helloworld:
	@python src/helloworld.py
