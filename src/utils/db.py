import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()

def get_engine() -> Engine:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "football_dw")
    user = os.getenv("DB_USER", "football")
    pwd = os.getenv("DB_PASSWORD", "football")

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url, pool_pre_ping=True)