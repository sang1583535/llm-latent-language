import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from utils import plot_ci

def load_means(raw_path):
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data['steps'] gives the x-axis labels
    layers = data['layers']
    # Convert nested lists to [n_prompts, n_steps] tensors for each layer
    means_en = {layer: [] for layer in layers}
    means_tgt = {layer: [] for layer in layers}
    for layer in layers:
        # per_layer_en_by_step[layer] is list over steps; each element is list over prompts
        per_step_en = data['per_layer_en_by_step'][str(layer)]
        per_step_tgt = data['per_layer_tgt_by_step'][str(layer)]
        # For each step, compute mean over prompts
        means_en[layer] = [np.mean(step_vals) for step_vals in per_step_en]
        means_tgt[layer] = [np.mean(step_vals) for step_vals in per_step_tgt]
    return data['steps'], means_en, means_tgt

# Load both models
steps1, en1, zh1 = load_means('/scratch/e1583535/llm-latent-language/visuals/results_01012026_3/para-only-7B/cloze_steps/zh/raw_dump.json')
steps2, en2, zh2 = load_means('/scratch/e1583535/llm-latent-language/visuals/results_01012026_3/multilingual-uniform-7B/cloze_steps/zh/raw_dump.json')
assert steps1 == steps2, "Steps must match to overlay."

# Choose a layer to plot, e.g. 10 or 31
layer = 31  # final block (32nd layer)

fig, ax = plt.subplots(figsize=(6,3))
# prepare tensors shaped [n_prompts=1, n_steps] because plot_ci expects 2D
en1_t = torch.tensor([en1[layer]], dtype=torch.float32)
zh1_t = torch.tensor([zh1[layer]], dtype=torch.float32)
en2_t = torch.tensor([en2[layer]], dtype=torch.float32)
zh2_t = torch.tensor([zh2[layer]], dtype=torch.float32)

plot_ci(ax, en1_t, "en model1", color="orange", tik_step=1, do_lines=False)
plot_ci(ax, zh1_t, "zh model1", color="blue", tik_step=1, do_lines=False)
plot_ci(ax, en2_t, "en model2", color="green", tik_step=1, do_lines=False)
plot_ci(ax, zh2_t, "zh model2", color="red", tik_step=1, do_lines=False)
ax.set_xlabel("training tokens")
ax.set_ylabel("probability")
# tighten the y-axis if probabilities are low
ax.set_ylim(0, 0.5)
# use the shared step labels (e.g. ["2B","4B",...] ) as x‐tick labels
ax.set_xticks(np.arange(1, len(steps1) + 1))
ax.set_xticklabels(steps1, rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("layer{}_comparison.png".format(layer))
