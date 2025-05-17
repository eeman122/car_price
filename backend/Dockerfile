FROM python:3.9-slim

WORKDIR /app

# 1. Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy requirements FIRST for caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Handle models (works for both local and HF)
COPY models.tar.gz .
RUN if [ -f models.tar.gz ]; then tar -xzvf models.tar.gz && rm models.tar.gz; fi

# 4. Copy application code
COPY . .

# Health check (essential for HF)
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Production settings (optimized for Hugging Face)
EXPOSE 8000
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30", \
     "--lifespan", "on"]