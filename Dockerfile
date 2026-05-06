FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "-lc", "python -c \"import api.app\" && uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000} --log-level debug"]