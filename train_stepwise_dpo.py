from data_utils import load_prm800k_dataset, preprocess_prm800k
from stepwise_dpo_trainer import StepwiseDPOTrainer
from data_collator import StepwiseDataCollator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

if __name__ == "__main__":
    # Load and preprocess data
    dataset = load_prm800k_dataset()
    train_processed = preprocess_prm800k(dataset["train"])

    # Load model and tokenizer (use a small model for testing)
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare data collator
    data_collator = StepwiseDataCollator()

    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
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
