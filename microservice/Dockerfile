FROM python:3.11-slim

WORKDIR /app

COPY microservice/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY microservice/ .

EXPOSE 8000

CMD ["python", "-m", "main"]
