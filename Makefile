install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	isort ml/*.py *.py
	black ml/*.py *.py

lint:
	flake8 ml/*.py *.py
	mypy ml/*.py *.py

build_data:
	python ml/1_build.py

train:
	python ml/2_train.py

test:
	python ml/3_test.py

retrain:
	python ml/6_retrain.py

pipeline: build_data train test

api:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t fashion-mnist-api .

docker-run:
	docker run -p 8000:8000 fashion-mnist-api

test-deployed:
	python ml/7_test_deployed_model.py

all: install format lint pipeline