FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p model data/processed data/splits

# Remove old mlflow db to avoid schema conflicts, then train
RUN rm -f mlflow.db mlflow_docker.db && \
    python scripts/preprocess.py && \
    python scripts/train_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]