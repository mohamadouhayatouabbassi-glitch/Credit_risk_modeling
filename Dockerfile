FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

<<<<<<< HEAD
CMD ["bash", "-lc", "python -m uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
=======
CMD ["bash", "-lc", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
>>>>>>> origin/main
