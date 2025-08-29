#!/usr/bin/env python3
"""
Script to quantize ONNX models to INT8 for faster inference.
"""

import os
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model():
    """Quantize the ONNX model to INT8."""
    # Paths
    model_dir = "./onnx_model"
    quantized_dir = "./onnx_model_quantized"
    
    # Create quantized directory
    os.makedirs(quantized_dir, exist_ok=True)
    
    print("Quantizing ONNX models to INT8...")
    
    # Quantize encoder model
    encoder_model_path = os.path.join(model_dir, "encoder_model.onnx")
    quantized_encoder_path = os.path.join(quantized_dir, "encoder_model_quantized.onnx")
    
    print(f"Quantizing encoder model: {encoder_model_path}")
    quantize_dynamic(
        model_input=encoder_model_path,
        model_output=quantized_encoder_path,
        weight_type=QuantType.QInt8
    )
    
    # Quantize decoder model
    decoder_model_path = os.path.join(model_dir, "decoder_model.onnx")
    quantized_decoder_path = os.path.join(quantized_dir, "decoder_model_quantized.onnx")
    
    print(f"Quantizing decoder model: {decoder_model_path}")
    quantize_dynamic(
        model_input=decoder_model_path,
        model_output=quantized_decoder_path,
        weight_type=QuantType.QInt8
    )
    
    # Copy tokenizer files
    tokenizer_files = [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    import shutil
    for file in tokenizer_files:
        src = os.path.join(model_dir, file)
        dst = os.path.join(quantized_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    print(f"Quantized models saved to {quantized_dir}")

def test_quantized_model():
    """Test the quantized model."""
    print("Testing quantized model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./onnx_model_quantized")
    
    # Load quantized ONNX models
    encoder_session = ort.InferenceSession("./onnx_model_quantized/encoder_model_quantized.onnx")
    decoder_session = ort.InferenceSession("./onnx_model_quantized/decoder_model_quantized.onnx")
    
    # Test input
    test_text = "răng bựa tê miềng nỏ chộ hắn hề?"
    print(f"Input text: {test_text}")
    
    # Tokenize input
    inputs = tokenizer(
        test_text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Run encoder
    encoder_outputs = encoder_session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
    )
    
    print("Encoder inference successful")
    
    # Initialize decoder input
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    
    # Run decoder
    decoder_outputs = decoder_session.run(
        None,
        {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_outputs[0],
            "encoder_attention_mask": inputs["attention_mask"].astype(np.int64)
        }
    )
    
    print("Decoder inference successful")
    
    # Decode output
    predicted_ids = np.argmax(decoder_outputs[0], axis=-1)
    normalized_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    
    print(f"Normalized text: {normalized_text}")

def compare_model_sizes():
    """Compare model sizes between original and quantized models."""
    print("Comparing model sizes...")
    
    # Original model sizes
    encoder_original_size = os.path.getsize("./onnx_model/encoder_model.onnx")
    decoder_original_size = os.path.getsize("./onnx_model/decoder_model.onnx")
    original_total = encoder_original_size + decoder_original_size
    
    # Quantized model sizes
    encoder_quantized_size = os.path.getsize("./onnx_model_quantized/encoder_model_quantized.onnx")
    decoder_quantized_size = os.path.getsize("./onnx_model_quantized/decoder_model_quantized.onnx")
    quantized_total = encoder_quantized_size + decoder_quantized_size
    
    print(f"Original models size: {original_total / (1024*1024):.2f} MB")
    print(f"Quantized models size: {quantized_total / (1024*1024):.2f} MB")
    print(f"Size reduction: {((original_total - quantized_total) / original_total) * 100:.2f}%")

if __name__ == "__main__":
    print("ONNX Model Quantization Tool")
    print("=" * 50)
    
    # Quantize models
    quantize_onnx_model()
    
    # Test quantized model
    test_quantized_model()
    
    # Compare sizes
    compare_model_sizes()
    
    print("\nQuantization completed successfully!")