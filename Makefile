SHELL := /bin/bash

APP_MODULE ?= app.main:app
HOST ?= 127.0.0.1
PORT ?= 8000
WORKERS ?= 2
RELOAD ?= false

.PHONY: help venv install dev prod public test health

help:
	@echo "Available targets:"
	@echo "  make venv      - Create local virtual environment"
	@echo "  make install   - Install dependencies into .venv"
	@echo "  make dev       - Run FastAPI in development mode"
	@echo "  make prod      - Run FastAPI for production behind Nginx"
	@echo "  make public    - Run FastAPI publicly (without Nginx)"
	@echo "  make health    - Check health endpoint"

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

dev:
	. .venv/bin/activate && uvicorn $(APP_MODULE) --host 127.0.0.1 --port $(PORT) --reload

prod:
	. .venv/bin/activate && uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT) --workers $(WORKERS)

public:
	. .venv/bin/activate && uvicorn $(APP_MODULE) --host 0.0.0.0 --port $(PORT) --workers $(WORKERS)

health:
	curl -fsS http://127.0.0.1:$(PORT)/health && echo
