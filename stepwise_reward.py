# stepwise_reward.py

"""
Implements the LLM-based stepwise reward model.
"""

from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class StepwiseRewardModel:
    """
    Uses a Hugging Face model to assign a reward score to each step in a solution.
    Replace the scoring logic with an LLM call for real use.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def evaluate_steps(self, steps: List[str]) -> List[float]:
        """
        Given a list of steps, return a reward score for each step.
        Currently returns a dummy score (to be replaced with LLM logic).
        """
        scores = []
        for step in steps:
            # Tokenize and get model output (placeholder logic)
            inputs = self.tokenizer(step, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = float(outputs.logits[0][0].item())  # Dummy: use first logit
            scores.append(score)
        return scores