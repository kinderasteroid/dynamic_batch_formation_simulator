FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Drop build-essential, cmake, and libopenblas-dev entirely.
# Pull the pre-compiled CUDA wheel directly. Change cu124 to match your host CUDA version (e.g., cu121, cu122) if needed.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    --only-binary llama-cpp-python

COPY . .

# Removed the mkdir command; Docker Compose automatically creates host directories when mapping volumes.
CMD ["python", "scripts/run_experiments.py"]