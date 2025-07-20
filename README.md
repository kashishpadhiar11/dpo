# Stepwise DPO Assignment

## Overview

This repository implements a stepwise Direct Preference Optimization (DPO) pipeline inspired by the methodology in ["Let's Verify Step by Step" (OpenAI)](https://arxiv.org/html/2408.15240v1). The goal is to use an LLM-based stepwise reward model to evaluate and improve solutions in a stepwise fashion, without relying on human annotators.

## Features

- **LLM-based stepwise reward model**: Uses a language model to assign rewards to each step in a solution.
- **Custom StepwiseDPOTrainer**: Subclasses Hugging Face’s `Trainer` to aggregate step-level rewards.
- **Reproducible experiments**: All code, requirements, and results are tracked for easy reproduction.
- **Active Git usage**: Frequent, descriptive commits document reasoning and experiment trails.

## How to Run

1. **Install requirements**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**  
   Place `train.jsonl` and `test.jsonl` in the `data/` directory.

3. **Preprocess data**  
   ```bash
   python data_utils.py
   ```

4. **Train the model**  
   ```bash
   python train_stepwise_dpo.py
   ```

## Results

- **Training loss**: See logs in the `results/` directory.
- **(Optional) Plots**: You can generate and view loss curves or other metrics in `results/`.
- **(Optional) Model checkpoints**: Saved in `results/` if enabled in the training script.

## Trade-offs & Next Steps

- Used a tiny model (`sshleifer/tiny-gpt2`) for local development due to hardware constraints. For real experiments, use a larger instruction-tuned model on a GPU machine or cloud environment.
- The current pipeline is modular and can be extended to support stepwise reward aggregation with more advanced models or datasets.
- Future work: Integrate rejected completions for contrastive training, add evaluation scripts, and experiment with larger models.

## LLM Usage

See [LLM_USAGE.md](./LLM_USAGE.md) for details on AI/code generation assistance.

## References

- ["Let's Verify Step by Step" (OpenAI)](https://arxiv.org/html/2408.15240v1)
- [Step-DPO Baseline Code](https://github.com/dvlab-research/Step-DPO)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

