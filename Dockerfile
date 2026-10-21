FROM node:22-alpine AS web-build

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/index.html web/vite.config.js ./
COPY web/src ./src
RUN npm run build

FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY firephantom ./firephantom
COPY --from=web-build /web/dist ./web/dist

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8765 \
    DATASET_ROOT=/data \
    DVC_NO_ANALYTICS=1

EXPOSE 8765

CMD ["python", "-m", "uvicorn", "firephantom.api:app", "--host", "0.0.0.0", "--port", "8765"]
