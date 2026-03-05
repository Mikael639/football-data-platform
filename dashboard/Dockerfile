FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir --upgrade pip
RUN useradd --create-home --uid 10001 appuser

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard /app/dashboard
COPY src /app/src
COPY data /app/data

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
