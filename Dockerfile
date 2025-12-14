FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ollama_client.py .
COPY quiz_service.py .
COPY api.py .
COPY main.py .

RUN mkdir -p /app/uploads /app/tmp

ENV OLLAMA_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3.2b
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_RELOAD=false
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "main.py"]