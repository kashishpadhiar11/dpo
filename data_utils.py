import os
from datasets import load_dataset

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

if __name__ == "__main__":
    dataset = load_prm800k_dataset()
    print(dataset)
    print("Sample train example:", dataset['train'][0])
