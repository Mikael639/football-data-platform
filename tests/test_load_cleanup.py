from sqlalchemy import create_engine, text

from src.load import cleanup_legacy_fact_rows_for_csv


def test_cleanup_legacy_fact_rows_for_csv_deletes_null_season_matches_and_stats():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE fact_match (match_id INTEGER PRIMARY KEY, season TEXT)"))
        conn.execute(
            text(
                """
                CREATE TABLE fact_player_match_stats (
                    match_id INTEGER,
                    player_id INTEGER,
                    PRIMARY KEY (match_id, player_id)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO fact_match (match_id, season)
                VALUES (1, NULL), (2, '2024-2025')
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO fact_player_match_stats (match_id, player_id)
                VALUES (1, 10), (2, 20)
                """
            )
        )

    counts = cleanup_legacy_fact_rows_for_csv(engine)

    assert counts == {"deleted_player_match_stats": 1, "deleted_matches": 1}
    with engine.begin() as conn:
        remaining_matches = conn.execute(text("SELECT COUNT(*) FROM fact_match")).scalar_one()
        remaining_stats = conn.execute(text("SELECT COUNT(*) FROM fact_player_match_stats")).scalar_one()
    assert remaining_matches == 1
    assert remaining_stats == 1
