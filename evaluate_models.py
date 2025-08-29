import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import onnxruntime
import os
import numpy as np
from peft import PeftModel
import pandas as pd
import jiwer


def get_model_size(path):
    if not os.path.isdir(path):
        return os.path.getsize(path) / (1024 * 1024)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


def load_pytorch_model():
    # Load the original PyTorch model.
    print("--- Loading Original PyTorch Model ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for PyTorch model")
    base_model_name = "VietAI/vit5-base"
    pytorch_model_path = "./qlora_best_model"
    pytorch_tokenizer = T5Tokenizer.from_pretrained(pytorch_model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    pytorch_model = PeftModel.from_pretrained(base_model, pytorch_model_path).to(device)
    return pytorch_model, pytorch_tokenizer


def get_onnx_providers():
    """Get available ONNX providers, preferring CUDA if available."""
    available_providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        print("Using GPU (CUDA) for ONNX inference")
        return [
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
        print("GPU (CUDA) not available, using CPU for ONNX inference")
        return ['CPUExecutionProvider']


def load_onnx_model():
    # Load the ONNX model.
    print("--- Loading ONNX Model ---")
    onnx_model_path = "onnx_model"
    onnx_tokenizer = T5Tokenizer.from_pretrained(onnx_model_path)
    providers = get_onnx_providers()
    encoder_session = onnxruntime.InferenceSession(
        os.path.join(onnx_model_path, "encoder_model.onnx"),
        providers=providers
    )
    decoder_session = onnxruntime.InferenceSession(
        os.path.join(onnx_model_path, "decoder_model.onnx"),
        providers=providers
    )
    return (encoder_session, decoder_session), onnx_tokenizer


def load_quantized_onnx_model():
    # Load the quantized ONNX model.
    print("--- Loading Quantized ONNX Model ---")
    quantized_model_path = "onnx_model_quantized"
    quantized_tokenizer = T5Tokenizer.from_pretrained(quantized_model_path)
    providers = get_onnx_providers()
    quantized_encoder_session = onnxruntime.InferenceSession(
        os.path.join(quantized_model_path, "encoder_model_quantized.onnx"),
        providers=providers
    )
    quantized_decoder_session = onnxruntime.InferenceSession(
        os.path.join(quantized_model_path, "decoder_model_quantized.onnx"),
        providers=providers
    )
    return (quantized_encoder_session, quantized_decoder_session), quantized_tokenizer


def generate_pytorch_output(model, tokenizer, text, max_length=50):
    # Generate output using the PyTorch model.
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = inputs.input_ids.to(model.device)
    outputs = model.generate(input_ids=input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_onnx_output(encoder_session, decoder_session, tokenizer, text, max_length=50):
    # Generate output using the ONNX model.
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = inputs.input_ids.numpy().astype(np.int64)
    attention_mask = inputs.attention_mask.numpy().astype(np.int64)
    
    # Encoder
    encoder_outputs = encoder_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    encoder_hidden_states = encoder_outputs[0]
    
    # Decoder
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    
    for _ in range(max_length):
        decoder_outputs = decoder_session.run(None, {
            "input_ids": decoder_input_ids, 
            "encoder_hidden_states": encoder_hidden_states, 
            "encoder_attention_mask": attention_mask
        })
        next_token_logits = decoder_outputs[0][0, -1, :]
        next_token_id = np.argmax(next_token_logits)
        if next_token_id == tokenizer.eos_token_id:
            break
        decoder_input_ids = np.concatenate([decoder_input_ids, [[next_token_id]]], axis=1)
    
    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)


def evaluate_models_on_test_set(test_file_path):
    #
    # Evaluate the original PyTorch model, the ONNX model, and the quantized ONNX model on the test set.
    #
    # Load models
    pytorch_model, pytorch_tokenizer = load_pytorch_model()
    onnx_model, onnx_tokenizer = load_onnx_model()
    quantized_onnx_model, quantized_onnx_tokenizer = load_quantized_onnx_model()
    
    # Load test data
    print("\n--- Loading Test Data ---")
    test_df = pd.read_csv(test_file_path)
    test_df = test_df.dropna() # Drop rows with missing values
    
    # Initialize lists to store results
    pytorch_outputs = []
    onnx_outputs = []
    quantized_onnx_outputs = []
    ground_truths = test_df['standard'].tolist()
    
    # Initialize timing lists
    pytorch_times = []
    onnx_times = []
    quantized_onnx_times = []
    
    print("\n--- Evaluating Models ---")
    for i, row in test_df.iterrows():
        dialect_text = row['dialect']
        standard_text = row['standard']
        
        # Evaluate PyTorch model
        start_time = time.time()
        pytorch_output = generate_pytorch_output(pytorch_model, pytorch_tokenizer, dialect_text)
        pytorch_time = time.time() - start_time
        pytorch_outputs.append(pytorch_output)
        pytorch_times.append(pytorch_time)
        
        # Evaluate ONNX model
        start_time = time.time()
        onnx_output = generate_onnx_output(onnx_model[0], onnx_model[1], onnx_tokenizer, dialect_text)
        onnx_time = time.time() - start_time
        onnx_outputs.append(onnx_output)
        onnx_times.append(onnx_time)
        
        # Evaluate Quantized ONNX model
        start_time = time.time()
        quantized_onnx_output = generate_onnx_output(quantized_onnx_model[0], quantized_onnx_model[1], quantized_onnx_tokenizer, dialect_text)
        quantized_onnx_time = time.time() - start_time
        quantized_onnx_outputs.append(quantized_onnx_output)
        quantized_onnx_times.append(quantized_onnx_time)
        
        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_df)} samples")
    
    # Calculate metrics
    print("\n--- Calculating Metrics ---")
    
    # Character Error Rate (CER) using jiwer
    pytorch_cer = jiwer.cer(ground_truths, pytorch_outputs)
    onnx_cer = jiwer.cer(ground_truths, onnx_outputs)
    quantized_onnx_cer = jiwer.cer(ground_truths, quantized_onnx_outputs)
    
    # Word Error Rate (WER) using jiwer
    pytorch_wer = jiwer.wer(ground_truths, pytorch_outputs)
    onnx_wer = jiwer.wer(ground_truths, onnx_outputs)
    quantized_onnx_wer = jiwer.wer(ground_truths, quantized_onnx_outputs)
    
    # Average inference times
    avg_pytorch_time = sum(pytorch_times) / len(pytorch_times)
    avg_onnx_time = sum(onnx_times) / len(onnx_times)
    avg_quantized_onnx_time = sum(quantized_onnx_times) / len(quantized_onnx_times)
    
    # Get model sizes
    pytorch_model_size = get_model_size("./qlora_best_model")
    onnx_model_size = get_model_size("onnx_model")
    quantized_onnx_model_size = get_model_size("onnx_model_quantized")
    
    # Print comparison table
    print("\n--- Model Evaluation on Test Set ---")
    print(f"{'Model':<25} | {'Avg Inference Time (s)':<25} | {'Model Size (MB)':<20} | {'CER':<10} | {'WER':<10}")
    print("-" * 100)
    print(f"{'PyTorch (Original)':<25} | {avg_pytorch_time:<25.4f} | {pytorch_model_size:<20.2f} | {pytorch_cer:<10.4f} | {pytorch_wer:<10.4f}")
    print(f"{'ONNX':<25} | {avg_onnx_time:<25.4f} | {onnx_model_size:<20.2f} | {onnx_cer:<10.4f} | {onnx_wer:<10.4f}")
    print(f"{'ONNX (Quantized)':<25} | {avg_quantized_onnx_time:<25.4f} | {quantized_onnx_model_size:<20.2f} | {quantized_onnx_cer:<10.4f} | {quantized_onnx_wer:<10.4f}")


if __name__ == "__main__":
    test_file_path = "data/test.csv"
    evaluate_models_on_test_set(test_file_path)