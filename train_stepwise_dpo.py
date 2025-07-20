from data_utils import load_prm800k_dataset, preprocess_prm800k
from stepwise_dpo_trainer import StepwiseDPOTrainer
from data_collator import StepwiseDataCollator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
torch.device("cpu")

if __name__ == "__main__":
    # Load and preprocess data
    dataset = load_prm800k_dataset()
    train_processed = preprocess_prm800k(dataset["train"])

    # Load model and tokenizer (use a small model for testing)
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare data collator
    data_collator = StepwiseDataCollator(tokenizer)

    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Prepare dataset for Trainer (as a Hugging Face Dataset)
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_processed[:100])  # Use a small subset for testing

    # Initialize trainer
    trainer = StepwiseDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()
