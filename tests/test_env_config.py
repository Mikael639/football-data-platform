from src.utils.db import build_database_url


def test_build_database_url_uses_defaults():
    assert build_database_url({}) == "postgresql+psycopg2://football:football@localhost:5432/football_dw"


def test_build_database_url_reads_env_mapping():
    env = {
        "DB_HOST": "postgres",
        "DB_PORT": "5433",
        "DB_NAME": "warehouse",
        "DB_USER": "etl",
        "DB_PASSWORD": "secret",
    }

    assert build_database_url(env) == "postgresql+psycopg2://etl:secret@postgres:5433/warehouse"
