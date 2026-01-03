FROM python:3.11-slim

WORKDIR /app

# Install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace exposes ONLY port 7860
EXPOSE 7860

# Run FastAPI on the correct port
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "7860"]
