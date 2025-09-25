# Base image Python
FROM python:3.12-slim

# Install dependencies di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    nodejs \
    npm \
    unzip \
    && rm -rf /var/lib/apt/lists/*


# Imposta cartella di lavoro
WORKDIR /workspace
# Installa librerie Python
RUN pip install --no-cache-dir  uv

COPY ./pyproject.toml .
# Copia requirements

RUN uv sync
# Default shell
CMD [ "bash" ]