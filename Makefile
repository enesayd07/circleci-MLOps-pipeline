install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	isort ml/*.py *.py
	black ml/*.py *.py

lint:
	flake8 ml/*.py *.py
	mypy ml/*.py *.py

build_data:
	python ml/build.py

train:
	python ml/train.py

test:
	python ml/test.py

retrain:
	python ml/retrain.py

pipeline: build_data train test

api:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t fashion-mnist-api .

docker-run:
	docker run -p 8000:8000 fashion-mnist-api

test-deployed:
	python ml/test_deployed_model.py

all: install format lint pipeline