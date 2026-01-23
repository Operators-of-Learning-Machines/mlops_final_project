FROM python:3.12-slim-bookworm

WORKDIR /app

COPY frontend/index.html ./index.html

EXPOSE 8080

ENTRYPOINT ["python3", "-m", "http.server", "8080", "--bind", "0.0.0.0"]
