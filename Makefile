install:
	@pip install -r ./requirements.txt

venv-create:
	@python -m venv .venv

venv-use:
	@echo "You need to run:"
	@echo "source .venv/bin/activate"

freeze-deps:
	@pip freeze > requirements.txt

1-helloworld:
	@python src/1-helloworld.py

2-load-dataset:
	@python src/2-load-dataset.py


3-build-model:
	@python src/3-build-model.py

4-text-classification:
	@echo "https://www.tensorflow.org/tutorials/keras/text_classification"
	@python src/4-text-classification.py

jupyter:
	@cd src/notebooks && jupyter notebook
