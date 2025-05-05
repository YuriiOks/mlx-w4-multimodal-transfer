# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create final application image
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and necessary utilities
COPY app/ app/
COPY src/ src/
COPY utils/ utils/
# COPY config.yaml . # Might mount this instead

# Download NLTK data if needed for BLEU score
# RUN python -m nltk.downloader punkt

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck (optional)
HEALTHCHECK CMD streamlit hello

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

