.PHONY: up down init run logs psql

up:
	docker compose up -d

down:
	docker compose down

init:
	docker cp sql/01_schema.sql football_postgres:/01_schema.sql
	docker exec -it football_postgres psql -U football -d football_dw -f /01_schema.sql
	docker cp sql/02_indexes.sql football_postgres:/02_indexes.sql
	docker exec -it football_postgres psql -U football -d football_dw -f /02_indexes.sql

run:
	python -m src.run_pipeline

logs:
	docker logs -f football_postgres

psql:
	docker exec -it football_postgres psql -U football -d football_dw