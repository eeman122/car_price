FROM python:3.9

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy everything into container
COPY . .

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose both ports
EXPOSE 7860 8000

# Health check for Hugging Face
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run both FastAPI and Streamlit
CMD ["bash", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py"]

