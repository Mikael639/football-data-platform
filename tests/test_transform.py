from src.extract import extract_from_mock
from src.transform import transform

def test_transform_shapes():
    payload = extract_from_mock()
    t = transform(payload)

    assert len(t["fact_match"]) == 2
    assert len(t["fact_player_match_stats"]) == 3

    # dim tables should not be empty
    assert len(t["dim_team"]) >= 2
    assert len(t["dim_player"]) >= 3
    assert len(t["dim_date"]) == 2