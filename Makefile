.PHONY: up down init migrate run study-fbref study-fbref-docker study-fbref-manual study-fbref-manual-docker fbref-standard-supabase-import fbref-standard-supabase-import-docker fbref-cleaned-to-manual-csv fbref-matchlogs-supabase-import fbref-matchlogs-supabase-import-docker logs psql

up:
	docker compose up -d

down:
	docker compose down

init:
	docker cp sql/01_schema.sql football_postgres:/01_schema.sql
	docker exec -it football_postgres psql -U football -d football_dw -f /01_schema.sql
	docker cp sql/02_indexes.sql football_postgres:/02_indexes.sql
	docker exec -it football_postgres psql -U football -d football_dw -f /02_indexes.sql
	docker cp sql/03_migrate_add_player_photo_url.sql football_postgres:/03_migrate_add_player_photo_url.sql
	docker exec -it football_postgres psql -U football -d football_dw -f /03_migrate_add_player_photo_url.sql

migrate:
	docker cp sql/03_migrate_add_player_photo_url.sql football_postgres:/03_migrate_add_player_photo_url.sql
	docker exec -it football_postgres psql -U football -d football_dw -f /03_migrate_add_player_photo_url.sql

run:
	python -m src.run_pipeline

study-fbref:
	python -m src.study_fbref

study-fbref-docker:
	docker compose run --rm pipeline python -m src.study_fbref

study-fbref-manual:
	$env:FBREF_STUDY_SOURCE="manual_csv"; python -m src.study_fbref

study-fbref-manual-docker:
	docker compose run --rm -e FBREF_STUDY_SOURCE=manual_csv pipeline python -m src.study_fbref

fbref-standard-supabase-import:
	python -m src.import_fbref_standard_to_supabase

fbref-standard-supabase-import-docker:
	docker compose run --rm pipeline python -m src.import_fbref_standard_to_supabase

fbref-cleaned-to-manual-csv:
	python -m src.build_player_match_manual_from_cleaned

fbref-matchlogs-supabase-import:
	python -m src.import_fbref_match_logs_to_supabase

fbref-matchlogs-supabase-import-docker:
	docker compose run --rm pipeline python -m src.import_fbref_match_logs_to_supabase

logs:
	docker logs -f football_postgres

psql:
	docker exec -it football_postgres psql -U football -d football_dw
