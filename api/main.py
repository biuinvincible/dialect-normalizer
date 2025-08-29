#!/usr/bin/env python3
"""
FastAPI application for Vietnamese dialect normalization using ONNX model.
Supports GPU execution and INT8 quantized models for faster inference.
"""

import os
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vietnamese Dialect Normalizer API",
    description="API for normalizing Vietnamese dialect text to standard Vietnamese using ONNX model with GPU and quantization support",
    version="1.2.0"
)

# Global variables for model and tokenizer
session = None
tokenizer = None
device = "cpu"  # Will be updated when model is initialized
model_type = "standard"  # Will be updated when model is initialized

class TextInput(BaseModel):
    text: str

class NormalizationOutput(BaseModel):
    input: str
    normalized: str

def get_device_info():
    """Get information about available execution providers."""
    available_providers = ort.get_available_providers()
    cuda_available = 'CUDAExecutionProvider' in available_providers
    return {
        "available_providers": available_providers,
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu"
    }

def initialize_model(use_quantized: bool = True):
    """Initialize the quantized ONNX model and tokenizer with GPU support and CPU fallback."""
    global session, tokenizer, device, model_type
    
    logger.info("Initializing model...")
    
    # Use quantized model directory
    model_dir = "./onnx_model_quantized"
    # Check if quantized model directory exists
    if not os.path.exists(model_dir):
        # Try alternative path (when running from api directory)
        model_dir = "../onnx_model_quantized"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                "Quantized ONNX model directory not found. Please run quantize_model.py first."
            )
    model_type = "quantized"
    
    # Use quantized model file names
    encoder_model_path = os.path.join(model_dir, "encoder_model_quantized.onnx")
    decoder_model_path = os.path.join(model_dir, "decoder_model_quantized.onnx")
    
    # Check if model files exist
    if not os.path.exists(encoder_model_path) or not os.path.exists(decoder_model_path):
        raise FileNotFoundError(
            f"Required quantized ONNX model files not found in {model_dir}. "
            f"Please run quantize_model.py first."
        )
    
    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    # Determine execution providers (GPU/CUDA first if available)
    device_info = get_device_info()
    device = device_info["device"]
    
    # Try to initialize with GPU first if available
    if device == "cuda":
        logger.info("CUDA is available, attempting to initialize model with GPU support")
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
    else:
        logger.info("CUDA is not available, using CPU execution")
        providers = ['CPUExecutionProvider']

    # Initialize ONNX Runtime sessions with selected providers
    try:
        logger.info(f"Attempting to initialize model with providers: {providers}")
        encoder_session = ort.InferenceSession(encoder_model_path, providers=providers)
        decoder_session = ort.InferenceSession(decoder_model_path, providers=providers)
        
        # Verify that the requested provider is being used
        encoder_provider = encoder_session.get_providers()[0]
        decoder_provider = decoder_session.get_providers()[0]
        
        if device == "cuda" and (encoder_provider != 'CUDAExecutionProvider' or decoder_provider != 'CUDAExecutionProvider'):
            logger.warning(f"Requested CUDA but model is running on {encoder_provider} and {decoder_provider}")
            device = "cpu"  # Update device to reflect actual execution provider
        else:
            device = "cuda" if encoder_provider == 'CUDAExecutionProvider' else "cpu"
            
        logger.info(f"Model successfully initialized with {encoder_provider} and {decoder_provider}")
    except Exception as e:
        logger.error(f"Error initializing model with {providers}: {e}")
        logger.info("Falling back to CPU execution")
        try:
            encoder_session = ort.InferenceSession(encoder_model_path, providers=['CPUExecutionProvider'])
            decoder_session = ort.InferenceSession(decoder_model_path, providers=['CPUExecutionProvider'])
            device = "cpu"
            logger.info("Model successfully initialized with CPU execution provider")
        except Exception as cpu_e:
            logger.error(f"Failed to initialize model even with CPU execution: {cpu_e}")
            raise RuntimeError(f"Failed to initialize model with any execution provider: {cpu_e}")
    
    session = {
        "encoder": encoder_session,
        "decoder": decoder_session
    }
    
    logger.info(f"Quantized ONNX model and tokenizer loaded successfully on {device.upper()}")

def generate_text(input_ids, attention_mask, max_length=128):
    """Generate text using the ONNX model with greedy decoding."""
    global session, tokenizer
    
    try:
        # Run encoder
        encoder_outputs = session["encoder"].run(
            None,
            {
                "input_ids": input_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64)
            }
        )
        
        # Initialize decoder input with bos token as start token if available, otherwise pad token
        start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
        decoder_input_ids = np.array([[start_token_id]], dtype=np.int64)
        
        # Generate tokens one by one
        for i in range(max_length):
            # Run decoder with current sequence
            decoder_outputs = session["decoder"].run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_outputs[0],
                    "encoder_attention_mask": attention_mask.astype(np.int64)
                }
            )
            
            # Get the next token (greedy decoding)
            # The logits are for the last position in the sequence
            next_token_logits = decoder_outputs[0][:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1)
            
            # Convert to scalar and then to array
            next_token_id_scalar = next_token_id.item()
            next_token_id_array = np.array([[next_token_id_scalar]], dtype=np.int64)
            
            # Check if we've generated an EOS token
            if next_token_id_scalar == tokenizer.eos_token_id:
                break
                
            # Append the next token to decoder input
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, next_token_id_array], 
                axis=1
            )
        
        # Decode the generated tokens, skipping special tokens
        generated_text = tokenizer.decode(
            decoder_input_ids[0], 
            skip_special_tokens=True
        )
        
        return generated_text
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize model when app starts."""
    logger.info("Starting up Vietnamese Dialect Normalizer API")
    try:
        # Initialize quantized model (required for production)
        initialize_model(use_quantized=True)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error: Could not initialize quantized model on startup: {e}")
        logger.error("API will start but model inference will fail until model is properly initialized")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    device_info = get_device_info()
    return {
        "message": "Vietnamese Dialect Normalizer API",
        "description": "API for normalizing Vietnamese dialect text to standard Vietnamese",
        "version": "1.2.0",
        "device_info": device_info,
        "model_type": model_type,
        "model_loaded": session is not None and tokenizer is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = session is not None and tokenizer is not None
    device_info = get_device_info()
    return {
        "status": "healthy" if model_loaded else "unhealthy", 
        "model_loaded": model_loaded,
        "device": device,
        "model_type": model_type,
        "device_info": device_info
    }

@app.post("/normalize", response_model=NormalizationOutput)
async def normalize_text(input_data: TextInput):
    """Normalize Vietnamese dialect text to standard Vietnamese."""
    global session, tokenizer
    
    # Check if model is loaded
    if session is None or tokenizer is None:
        logger.warning("Model not initialized, attempting to initialize now")
        try:
            # Initialize quantized model (required for production)
            initialize_model(use_quantized=True)
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
    
    try:
        # Tokenize input text
        inputs = tokenizer(
            input_data.text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate normalized text
        normalized_text = generate_text(
            inputs["input_ids"], 
            inputs["attention_mask"]
        )
        
        return NormalizationOutput(
            input=input_data.text,
            normalized=normalized_text
        )
    except Exception as e:
        logger.error(f"Error during normalization: {e}")
        raise HTTPException(status_code=500, detail=f"Error during normalization: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)