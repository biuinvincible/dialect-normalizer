# Stage 1: Builder stage to prepare the model
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04 AS builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
        gcc && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install Python dependencies for model preparation
COPY requirements-model.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements-model.txt

# Copy only the necessary files for model export and quantization
COPY export_onnx.py .
COPY quantize_model.py .
# Copy the trained model
COPY qlora_best_model/ ./qlora_best_model/

# Run model export and quantization
RUN python3.11 export_onnx.py
RUN python3.11 quantize_model.py


# Stage 2: Final stage for the API runtime
FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Create a non-root user
RUN useradd --create-home appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Ensure CUDA libraries are in the path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python and API dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
        curl \
        # Install cuDNN 9, required by ONNX Runtime GPU
        libcudnn9-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install API dependencies
COPY --chown=appuser:appuser requirements-api.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements-api.txt

# Copy API code and quantized model from builder stage
COPY --chown=appuser:appuser api/ ./api/
COPY --from=builder /app/onnx_model_quantized/ ./onnx_model_quantized/

# Switch to the non-root user
USER appuser

# Expose port
EXPOSE 8000

# Healthcheck with more detailed information
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
