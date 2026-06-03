import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import torch
import os

MODEL_NAME = "t5-small"
OUTPUT_DIR = "t5_finetuned_os"
CSV_FILE = "dataset_os_english.csv"
BATCH_SIZE = 8          
EPOCHS = 60             
LEARNING_RATE = 1e-3    

def run_training():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: File {CSV_FILE} Not Found!")
        return

    print(f"🚀 Starting The Fine-Tuning {MODEL_NAME}")

    # Load Data
    df = pd.read_csv(CSV_FILE)
    df = df.dropna()
    df['input_text'] = df['input_text'].astype(str)
    df['output_text'] = df['output_text'].astype(str)
    
    dataset = Dataset.from_pandas(df)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

    def preprocess_function(examples):
        
        inputs = ["question: " + doc for doc in examples["input_text"]]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        
        labels = tokenizer(
            text_target=examples["output_text"], 
            max_length=128, 
            truncation=True, 
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("⚙️  Tokenizing data...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    args = Seq2SeqTrainingArguments(
        output_dir="./results_temp",
        eval_strategy="no",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        weight_decay=0.00,       
        save_total_limit=1,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=False,
        use_cpu=not torch.cuda.is_available()
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("🔥 Start Training...")
    trainer.train()

    print(f"💾 Save to folder {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Model Trained Successfully")

if __name__ == "__main__":
    run_training()