FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 if needed (slim usually ok with binary, but keep minimal)
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY sql ./sql
COPY data ./data

# Default command: run pipeline
CMD ["python", "-m", "src.run_pipeline"]