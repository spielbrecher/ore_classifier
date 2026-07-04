.PHONY: build run stop clean

build:
	docker-compose build

run:
	docker-compose up

stop:
	docker-compose down

clean:
	docker-compose down -v
	docker system prune -f

train:
	python rag_classification.py

dev:
	streamlit run app.py