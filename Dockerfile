# Use the official Python 3.11 slim image (Debian-based, supports pre-compiled wheels)
FROM python:3.11-slim

# Prevent Python from writing pyc files and enforce unbuffered stdout for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required to build llama-cpp-python and its C/C++ backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies. 
# NOTE: If you are running this on a machine with an NVIDIA GPU, you should change 
# the pip install command to:
# RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the simulator source code into the container
COPY . .

# Ensure the results directory exists inside the container before running
RUN mkdir -p /app/results

# The default command runs the benchmarking script
CMD ["python", "scripts/run_experiments.py"]