from typing import List, Dict, Any

class StepwiseDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Concatenate question and chosen steps for each example
        texts = [
            example["question"] + "\n" + "\n".join(example["chosen_steps"])
            for example in batch
        ]
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # For language modeling, labels are usually the same as input_ids
        encodings["labels"] = encodings["input_ids"].clone()
        return encodings
