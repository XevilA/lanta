import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import numpy as np
import evaluate
import os

# Model selection - เลือกโมเดลที่เหมาะกับ Medical Translation
MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"  # Base model ที่ดีสำหรับ Chinese
MAX_LENGTH = 128  # ความยาวสูงสุดของประโยค

def preprocess_function(examples, tokenizer):
    """Tokenize และเตรียมข้อมูล"""
    inputs = examples["chinese"]
    targets = examples["thai"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_LENGTH, 
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=MAX_LENGTH, 
            truncation=True,
            padding="max_length"
        )
    
    # แปลง padding token เป็น -100 เพื่อไม่ให้คำนวณ loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer, metric):
    """คำนวณ BLEU Score"""
    preds, labels = eval_preds
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # คำนวณ BLEU
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def main():
    # Load dataset
    print("กำลังโหลด Dataset...")
    dataset = load_from_disk("datasets/chinese_thai_medical")
    
    # Load model และ tokenizer
    print(f"กำลังโหลดโมเดล {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # เพิ่ม special tokens สำหรับภาษาไทย
    special_tokens = ["<thai>", "</thai>", "<medical>", "</medical>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    # Tokenize dataset
    print("กำลัง Tokenize ข้อมูล...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Metric
    metric = evaluate.load("sacrebleu")
    
    # Training arguments - Optimized for 2-3 hours training
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        learning_rate=5e-5,
        per_device_train_batch_size=32,  # ปรับตาม GPU memory
        per_device_eval_batch_size=32,
        num_train_epochs=3,  # ปรับให้เหมาะกับเวลา 2-3 ชม
        weight_decay=0.01,
        warmup_steps=500,
        fp16=True,  # เร่งการเทรน
        gradient_accumulation_steps=2,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",  # ไม่ใช้ wandb
        dataloader_num_workers=4,
        logging_dir="./logs",
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("เริ่มการเทรน...")
    trainer.train()
    
    # Save final model
    print("บันทึกโมเดล...")
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    # Evaluate on test set
    print("ประเมินผลบน Test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(f"Test BLEU Score: {test_results['eval_bleu']:.2f}")
    
    # Save results
    import json
    with open("./final_model/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("เทรนเสร็จสมบูรณ์!")

if __name__ == "__main__":
    main()
