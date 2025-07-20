from typing import List, Dict, Any

class StepwiseDataCollator:
    """
    Simple data collator for stepwise DPO.
    """
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This example just returns the batch as-is.
        # You can add tokenization or batching logic here as needed.
        questions = [item["question"] for item in batch]
        chosen_steps = [item["chosen_steps"] for item in batch]
        return {
            "questions": questions,
            "chosen_steps": chosen_steps,
        }
