from data_utils import load_prm800k_dataset, preprocess_prm800k
from stepwise_reward.py import StepwiseRewardModel

if __name__ == "__main__":
    dataset = load_prm800k_dataset()
    train_processed = preprocess_prm800k(dataset["train"])
    reward_model = StepwiseRewardModel()

    # Score the first example's steps
    example = train_processed[0]
    print("Question:", example["question"])
    print("Steps:", example["chosen_steps"])
    scores = reward_model.evaluate_steps(example["chosen_steps"])
    print("Stepwise scores:", scores)
