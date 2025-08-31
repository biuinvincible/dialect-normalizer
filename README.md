# Vietnamese Dialect Normalizer ONNX Deployment Pipeline

This repository contains a complete MLOps pipeline for training, optimizing, and deploying a Vietnamese dialect normalization model. The project fine-tunes a `ViT5-base` model using QLoRA, converts it to the ONNX format, applies INT8 quantization, and serves it via a GPU-accelerated FastAPI application containerized with Docker.

## Key Features

- **QLoRA Training**: Efficiently fine-tunes the model using 4-bit quantization to reduce memory usage.
- **ONNX Optimization**: Exports the model to ONNX for high-performance, cross-platform inference.
- **INT8 Quantization**: Reduces model size by ~75% and speeds up inference with a minimal trade-off in accuracy.
- **GPU Acceleration**: The FastAPI application automatically utilizes available NVIDIA GPUs via ONNX Runtime, with a graceful fallback to CPU.
- **Dockerized Deployment**: A multi-stage `Dockerfile` creates a minimal, secure, and efficient runtime image.
- **CLI Client**: Includes an interactive command-line client for easy testing of the deployed API.

## Prerequisites

- Python 3.11.13
- Docker Engine
- **NVIDIA Container Toolkit**: Required to enable GPU support within Docker containers.
- A CUDA-compatible NVIDIA GPU with up-to-date drivers.

## End-to-End Guide

This guide walks through every step, from preparing the data to interacting with the final deployed API.

### Step 1: Setup

Clone the repository and install the required Python dependencies.

```bash
git clone https://github.com/biuinvincible/dialect-normalizer.git
cd vidialect
pip install -r requirements.txt
```

### Step 2: Data Preparation

Place your training, validation, and test data in the `data/` directory. The files must be in CSV format with two columns: `dialect` (the input) and `standard` (the desired output).

- `data/train.csv`
- `data/dev.csv`
- `data/test.csv`

### Step 3: Run the MLOps Pipeline

Execute the following scripts in order to train the model, export it to ONNX, and apply quantization.

1.  **Train the Model (QLoRA)**
    This saves the adapter weights to the `qlora_best_model/` directory.
    ```bash
    python train_model.py
    ```

2.  **Export to ONNX**
    This merges the adapter and base model and saves the ONNX model to `onnx_model/`.
    ```bash
    python export_onnx.py
    ```

3.  **Quantize the ONNX Model (INT8)**
    This creates the final, optimized model in the `onnx_model_quantized/` directory.
    ```bash
    python quantize_model.py
    ```

### Step 4: Build and Run the Docker Container

Build the Docker image and run the containerized API.

1.  **Build the Image**
    ```bash
    docker build -t dialect-normalizer-onnx .
    ```

2.  **Run the Container**
    Use the `--gpus all` flag to provide the container with access to your host's NVIDIA GPU.

    *   **With GPU Acceleration (Recommended):**
        ```bash
        docker run -d -p 8000:8000 --gpus all --name normalizer-api dialect-normalizer-onnx
        ```

    *   **With CPU Only:**
        ```bash
        docker run -d -p 8000:8000 --name normalizer-api dialect-normalizer-onnx
        ```

The API will now be accessible at `http://localhost:8000`.

### Step 5: Interact with the API

You can interact with the running API in two ways:

#### A) Using the CLI Client

Run the provided command-line client for an interactive session. Type your text, press Enter, and receive the normalized output. Type `quit` or `exit` to stop.

```bash
python cli_client.py
```

#### B) Using cURL

Send a POST request directly to the `/normalize` endpoint.

**Request:**
```bash
curl -X POST "http://localhost:8000/normalize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Bựa miềng đi chự bò bơ gặp hấn bên tê"}'
```

**Response:**
```json
{
  "input": "Bựa miềng đi chự bò bơ gặp hấn bên tê",
  "normalized": "Bữa mình đi giữ bò thì gặp nó bên kia"
}
```

### Using Docker Compose (Alternative)

For a simpler deployment, you can use Docker Compose, which reads the `docker-compose.yml` file to manage the container.

```bash
# Build and start the service in the background
docker-compose up --build -d

# Stop and remove the service
docker-compose down
```

## Troubleshooting

### Docker GPU Issues
If the GPU is not detected inside the container, ensure:
1.  You have installed the **NVIDIA Container Toolkit** on your host machine.
2.  Your Docker daemon is configured to use the NVIDIA runtime.
3.  You are using the `--gpus all` flag with `docker run`.

### Model Loading Issues
If the API reports that models have failed to load, ensure you have successfully run `train_model.py`, `export_onnx.py`, and `quantize_model.py` in order, and that the `onnx_model_quantized/` directory is correctly populated before building the Docker image.
