import os
from datasets import load_dataset
import json

def load_prm800k_dataset(data_dir: str = "data"):
    """
    Loads the prm800k dataset from local JSONL files using Hugging Face datasets.
    """
    data_files = {
        "train": os.path.join(data_dir, "/Users/bizzlemuffinn/Desktop/dpo/data/phase1_train.jsonl"),
        "test": os.path.join(data_dir, "/Users/bizzlemuffinn/Desktop/dpo/data/phase1_test.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)
    return dataset

def preprocess_prm800k(dataset):
    """
    Preprocesses the prm800k dataset to extract question and chosen stepwise solution.
    Returns a list of dicts with 'question' and 'chosen_steps'.
    """
    processed = []
    for example in dataset:
        # Get the question text
        question = None
        if "question" in example and isinstance(example["question"], dict):
            question = example["question"].get("problem")
        if not question:
            continue

        # Get the stepwise chosen solution
        steps = []
        label = example.get("label", {})
        for step in label.get("steps", []):
            completions = step.get("completions", [])
            chosen_idx = step.get("chosen_completion")
            if completions and chosen_idx is not None and 0 <= chosen_idx < len(completions):
                chosen_text = completions[chosen_idx].get("text")
                if chosen_text:
                    steps.append(chosen_text.strip())
        if not steps:
            continue

        processed.append({
            "question": question,
            "chosen_steps": steps,
        })
    return processed

if __name__ == "__main__":
    dataset = load_prm800k_dataset()
    train_processed = preprocess_prm800k(dataset["train"])
    print("First processed example:", train_processed[0] if train_processed else "No valid examples found.")
    print(f"Total processed examples: {len(train_processed)}")
    # Save to file
    with open("data/train_processed.jsonl", "w") as f:
        for item in train_processed:
            f.write(json.dumps(item) + "\n")
