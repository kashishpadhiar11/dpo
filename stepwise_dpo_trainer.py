from typing import Any, Dict
from transformers import Trainer

class StepwiseDPOTrainer(Trainer):
    """
    Custom trainer for Stepwise DPO, aggregating step-level rewards.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add any custom initialization here

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override to aggregate stepwise rewards.
        """
        # Example: Suppose 'inputs' contains a list of stepwise rewards
        # You would aggregate them here (e.g., sum, mean, etc.)
        # This is a placeholder; you will need to adapt it to your data format
        stepwise_rewards = inputs.get("stepwise_rewards", None)
        if stepwise_rewards is not None:
            # Aggregate rewards (e.g., sum)
            total_reward = sum(stepwise_rewards)
            # Use total_reward in your loss calculation
            # ...
        # Call the original compute_loss for now
        return super().compute_loss(model, inputs, return_outputs)