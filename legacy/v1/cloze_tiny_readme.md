# cloze_step.py

Evaluates how well a language model predicts target language words vs. English words across different training checkpoints using a cloze task.

## Process

1. **Build cloze dataset** — Creates few-shot prompts with masked translations:
   - Loads word pairs (English → target language translation)
   - Samples N demonstration examples as context
   - Query format: `<lang>: "` (open quote for word completion)

2. **For each checkpoint:**
   - Loads model at that training step
   - Runs each prompt through all layers
   - Extracts hidden states and final logits
   - Computes probability of English word vs. target word at final position

3. **Aggregate statistics:**
   - Collects mean and std of probabilities per layer
   - Calculates 95% confidence intervals using Gaussian approximation

4. **Output:**
   - Plots probability trends across training steps (per layer)
   - Saves `summary.json` with raw numbers
   - Saves `dataset.csv` with all prompts used

## Key outputs
- `layer_*.png` — Plot showing English vs. target language probability by training step
- `summary.json` — Raw probabilities and std for each layer and step
