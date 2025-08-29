#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
import evaluate
import torch
from jiwer import wer as calculate_wer
import editdistance
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

def main():
    # Download necessary NLTK resources
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # Load và xử lý dữ liệu
    train_df = pd.read_csv("data/train.csv")
    train_df = train_df.dropna()
    valid_df = pd.read_csv("data/dev.csv")
    valid_df = valid_df.dropna()
    test_df = pd.read_csv("data/test.csv")
    test_df = test_df.dropna()

    # Convert sang định dạng HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    # ================== QLoRA Configuration ==================
    # Cấu hình quantization (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # Load model in 4-bit
        bnb_4bit_use_double_quant=True,       # Use double quantization
        bnb_4bit_quant_type="nf4",            # Use NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype for 4-bit base models
    )

    # Load model với quantization config
    model_name = "VietAI/vit5-base"  # Use base model as requested
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model với 4-bit quantization
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically map model to available devices
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # ================== LoRA Configuration ==================
    # Cấu hình LoRA
    lora_config = LoraConfig(
        r=16,                                  # LoRA rank (càng cao càng nhiều params)
        lora_alpha=32,                         # LoRA scaling parameter
        target_modules=[                       # Target modules to apply LoRA
            "q",                               # Query projection
            "v",                               # Value projection
            "k",                               # Key projection
            "o",                               # Output projection
            "wi_0",                            # FFN input projection (T5 specific)
            "wi_1",                            # FFN input projection (T5 specific)
            "wo",                              # FFN output projection
            "lm_head",                         # Language model head
        ],
        lora_dropout=0.1,                      # LoRA dropout
        bias="none",                           # Bias training strategy
        task_type=TaskType.SEQ_2_SEQ_LM,      # Task type for seq2seq
        inference_mode=False,                  # Training mode
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters info
    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable %: {100 * trainable_params / all_param:.2f}"
        )

    print_trainable_parameters(model)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # ================== Data Preprocessing ==================
    # Tiền xử lý dữ liệu
    max_length = 50  # Điều chỉnh theo độ dài văn bản

    def preprocess_function(examples):
        inputs = examples["dialect"]
        targets = examples["standard"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )

        # For mBART or similar models
        if hasattr(tokenizer, 'src_lang'):
            tokenizer.src_lang = "vi_VN"
            tokenizer.tgt_lang = "vi_VN"

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Áp dụng tiền xử lý
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_valid = valid_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # Thiết lập data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ================== Metrics Configuration ==================
    # Load metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    # Định nghĩa hàm tính toán metrics
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 with the pad token ID before decoding
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Now decode both sequences
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = {}

        # Calculate ROUGE metrics
        try:
            rouge_result = rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            result["rouge_l"] = rouge_result["rougeL"]
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            result["rouge_l"] = float('nan')

        # Calculate BLEU
        try:
            bleu_result = bleu.compute(
                predictions=decoded_preds,
                references=decoded_labels
            )
            result["bleu"] = bleu_result["bleu"]
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            try:
                bleu_scores = []
                smoothing_function = SmoothingFunction().method1
                for pred, label in zip(decoded_preds, decoded_labels):
                    pred_tokens = pred.split()
                    label_tokens = label.split()
                    if len(label_tokens) > 0:
                        score = sentence_bleu([label_tokens], pred_tokens, smoothing_function=smoothing_function)
                        bleu_scores.append(score)
                result["bleu"] = np.mean(bleu_scores) if bleu_scores else float('nan')
            except Exception as e2:
                print(f"Manual BLEU calculation also failed: {e2}")
                result["bleu"] = float('nan')

        # Calculate METEOR
        try:
            meteor_scores = []
            for pred, label in zip(decoded_preds, decoded_labels):
                pred_tokens = pred.split()
                label_tokens = label.split()
                if len(label_tokens) > 0:
                    meteor_scores.append(meteor_score([label_tokens], pred_tokens))
            result["meteor"] = np.mean(meteor_scores) if meteor_scores else float('nan')
        except Exception as e:
            print(f"METEOR calculation error: {e}")
            result["meteor"] = float('nan')

        # Calculate WER
        try:
            valid_pairs = [(ref, pred) for ref, pred in zip(decoded_labels, decoded_preds) if len(ref.strip()) > 0]
            if valid_pairs:
                valid_refs, valid_preds = zip(*valid_pairs)
                wer_scores = [calculate_wer(ref, pred) for ref, pred in zip(valid_refs, valid_preds)]
                result["wer"] = np.mean(wer_scores)
            else:
                result["wer"] = float('nan')
        except Exception as e:
            print(f"WER calculation error: {e}")
            result["wer"] = float('nan')

        # Calculate CER
        def calculate_cer(ref, pred):
            if len(ref) == 0:
                return 1.0 if len(pred) > 0 else 0.0
            return editdistance.eval(ref, pred) / max(len(ref), 1)

        try:
            cer_scores = [calculate_cer(ref, pred) for ref, pred in zip(decoded_labels, decoded_preds)]
            result["cer"] = np.mean(cer_scores)
        except Exception as e:
            print(f"CER calculation error: {e}")
            result["cer"] = float('nan')

        return {k: round(v, 4) if not isinstance(v, float) or not np.isnan(v) else v for k, v in result.items()}

    # ================== Training Configuration with QLoRA ==================
    # Training arguments optimized for QLoRA
    training_args = Seq2SeqTrainingArguments(
        output_dir="./qlora_results",
        eval_strategy="epoch",
        learning_rate=1e-4,                    # Higher LR for LoRA
        per_device_train_batch_size=16,        # Smaller batch size due to memory
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,         # Accumulate gradients for effective batch size
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=15,
        predict_with_generate=True,
        fp16=False,                            # Use bf16 instead for better stability
        bf16=True,                             # Better for QLoRA
        logging_steps=50,
        report_to="none",
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        eval_accumulation_steps=1,
        save_steps=500,
        logging_first_step=True,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",             # 8-bit optimizer for memory efficiency
        warmup_ratio=0.03,                     # Warmup for stable training
        gradient_checkpointing=True,           # Save memory during backprop
        max_grad_norm=0.3,                     # Gradient clipping for stability
    )

    # Initialize Early Stopping Callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=1,
        early_stopping_threshold=0.001
    )

    # ================== Initialize Trainer ==================
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # ================== Training ==================
    print("Starting QLoRA training with early stopping...")
    print(f"Maximum epochs: {training_args.num_train_epochs}")
    print(f"Early stopping patience: {early_stopping_callback.early_stopping_patience}")
    print(f"Monitoring metric: {training_args.metric_for_best_model}")
    print(f"LoRA rank: {lora_config.r}")
    print(f"LoRA alpha: {lora_config.lora_alpha}")

    # Train the model
    trainer.train()

    # ================== Evaluation ==================
    # Evaluate on test set
    test_results = trainer.evaluate(tokenized_test, metric_key_prefix="test")
    print(f"Test results: {test_results}")

    # ================== Save Model ==================
    # Save LoRA adapter weights
    model_save_path = "./qlora_best_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"QLoRA model saved to {model_save_path}")

    # ================== Inference Functions ==================
    def normalize_text(text, max_length=50):
        """Normalize Vietnamese dialect text to standard Vietnamese"""
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate prediction
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_length)

        # Decode prediction
        normalized_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return normalized_text

    def translate(text, max_length=128):
        """Translate Vietnamese dialect to standard Vietnamese"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

        # Remove token_type_ids if exists
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Test inference
    test_text = "Bựa miềng đi chự bò bơ gặp hấn bên tê"
    print(f"\nTest inference:")
    print(f"Input: {test_text}")
    print(f"Output: {translate(test_text)}")

    # ================== Load Saved Model for Inference ==================
    def load_qlora_model(model_path, base_model_name):
        """Load QLoRA model for inference"""
        from peft import PeftModel

        # Load base model with quantization
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)

        # Merge LoRA weights for faster inference (optional)
        # model = model.merge_and_unload()

        return model

if __name__ == "__main__":
    main()