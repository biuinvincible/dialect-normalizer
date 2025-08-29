#!/usr/bin/env python3
"""
Export the Vietnamese dialect normalizer model to ONNX format.
This script handles the QLoRA adapter model by first loading the base model
and then merging the adapter weights.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, LoraConfig
from optimum.exporters.onnx import main_export

def export_model_to_onnx():
    # Model paths
    trained_model_path = "qlora_best_model"  # Path to our locally trained model
    base_model_name = "VietAI/vit5-base"    # Base model we used for training
    
    # Output directory for ONNX model
    output_dir = "onnx_model"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading base model {base_model_name}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    
    print(f"Loading trained adapter model from {trained_model_path}...")
    # Load our trained LoRA adapter weights
    model = PeftModel.from_pretrained(base_model, trained_model_path)
    
    print("Merging adapter weights with base model...")
    model = model.merge_and_unload()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    
    print("Saving merged model and tokenizer...")
    model.save_pretrained("temp_model")
    tokenizer.save_pretrained("temp_model")
    
    print("Exporting to ONNX format...")
    # Convert to ONNX using optimum with opset 14 to support triu operator
    main_export(
        model_name_or_path="temp_model",
        output=output_dir,
        task="seq2seq-lm",
        fp16=False,
        device="cpu",
        opset=14  # Use opset 14 to support triu operator
    )
    
    # Clean up temporary model
    import shutil
    shutil.rmtree("temp_model")
    
    print(f"Model successfully exported to {output_dir}")
    
    # List files in output directory
    print("Files created:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    export_model_to_onnx()