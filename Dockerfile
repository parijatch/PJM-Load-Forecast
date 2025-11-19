# Lightweight Python base image for linux/amd64
FROM python:3.11-slim

# Install system tools: make (for the Makefile) and a few basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
 && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (respecting .dockerignore)
COPY . .

# Default command: open a bash shell (for interactive runs)
CMD ["/bin/bash"]

