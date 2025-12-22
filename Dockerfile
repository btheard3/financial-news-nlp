FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY streamlit_app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy app + data (we need data/raw inside container)
COPY streamlit_app /app/streamlit_app
COPY data /app/data

EXPOSE 8080

CMD ["python", "-m", "streamlit", "run", "streamlit_app/app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
