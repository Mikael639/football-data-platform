import json
from pathlib import Path
from sqlalchemy import text

from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger("enrich_photos")

PHOTOS_PATH = Path("data/player_photos.json")

def main():
    if not PHOTOS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PHOTOS_PATH}")

    photos = json.loads(PHOTOS_PATH.read_text(encoding="utf-8"))
    engine = get_engine()

    updated = 0
    with engine.begin() as conn:
        for name, url in photos.items():
            res = conn.execute(
                text("""
                    UPDATE dim_player
                    SET photo_url = :url
                    WHERE full_name = :name
                """),
                {"url": url, "name": name},
            )
            updated += res.rowcount

    logger.info(f"Updated photo_url for {updated} players")

if __name__ == "__main__":
    main()