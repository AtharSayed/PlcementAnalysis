# Use lighter-weight Python 3.10 image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for matplotlib/Seaborn
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (except those in .dockerignore)
COPY . .

# Ensure data files are accessible
RUN chmod a+r data/*.csv

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "main.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]