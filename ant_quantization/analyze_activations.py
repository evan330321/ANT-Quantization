"""
Outlier Analysis Script for BERT-base ACTIVATIONS
==================================================
Analyzes activation tensors of BERT-base by running real inference
on SST-2 data, capturing outputs at every Linear layer.

Usage:
    python analyze_activations.py

Output:
    - activation_analysis/summary.csv     : per-layer activation statistics
    - activation_analysis/overall.txt     : aggregate statistics
    - activation_analysis/plots/*.png     : visualizations
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

# ============================================================
# Config
# ============================================================
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./activation_analysis"
SIGMA_THRESHOLD = 3.0
GROUP_SIZES = [2, 4, 8]
NUM_SAMPLES = 32       # how many SST-2 samples to feed through the model
MAX_LEN = 128

# ============================================================
# Setup
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ============================================================
# Hook: capture activation at every Linear layer
# ============================================================
activations = {}  # {layer_name: tensor}

def make_hook(name):
    def hook(module, input, output):
        # input to a Linear is a tuple (x,); we care about the input (activation)
        # store on CPU to save GPU memory
        act = input[0].detach().cpu()
        if name not in activations:
            activations[name] = []
        activations[name].append(act)
    return hook

hooks = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

print(f"Registered hooks on {len(hooks)} Linear layers")

# ============================================================
# Feed data through the model
# ============================================================
print(f"\nLoading SST-2 ({NUM_SAMPLES} samples)...")
dataset = load_dataset("glue", "sst2", split=f"validation[:{NUM_SAMPLES}]")

print("Running inference to collect activations...")
with torch.no_grad():
    for i, sample in enumerate(dataset):
        inputs = tokenizer(
            sample['sentence'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        ).to(device)
        _ = model(**inputs)
        if (i+1) % 8 == 0:
            print(f"  processed {i+1}/{NUM_SAMPLES}")

# Remove hooks
for h in hooks:
    h.remove()

# Concatenate all captured activations per layer
print("\nConcatenating activations...")
for name in list(activations.keys()):
    activations[name] = torch.cat(activations[name], dim=0)

# ============================================================
# Analysis functions (same as weight version, adapted)
# ============================================================
def analyze_tensor(tensor, name, sigma_threshold=3.0):
    flat = tensor.flatten().float().numpy()
    mean = flat.mean()
    std = flat.std()
    abs_max = np.abs(flat).max()
    max_sigma = abs_max / std if std > 0 else 0

    threshold = sigma_threshold * std
    outlier_mask = np.abs(flat - mean) > threshold
    n_outliers = outlier_mask.sum()
    outlier_ratio = n_outliers / len(flat)

    if n_outliers > 0:
        outlier_values = np.abs(flat[outlier_mask] - mean) / std
        dist_3_6   = ((outlier_values >= 3)   & (outlier_values < 6)).sum()   / n_outliers
        dist_6_10  = ((outlier_values >= 6)   & (outlier_values < 10)).sum()  / n_outliers
        dist_10_50 = ((outlier_values >= 10)  & (outlier_values < 50)).sum()  / n_outliers
        dist_50p   = (outlier_values >= 50).sum()                             / n_outliers
    else:
        dist_3_6 = dist_6_10 = dist_10_50 = dist_50p = 0.0

    return {
        'name': name,
        'shape': tuple(tensor.shape),
        'n_elements': len(flat),
        'mean': mean,
        'std': std,
        'abs_max': abs_max,
        'max_sigma': max_sigma,
        'n_outliers': int(n_outliers),
        'outlier_ratio_pct': outlier_ratio * 100,
        'dist_3_6_pct':   dist_3_6   * 100,
        'dist_6_10_pct':  dist_6_10  * 100,
        'dist_10_50_pct': dist_10_50 * 100,
        'dist_50p_pct':   dist_50p   * 100,
        'outlier_mask': outlier_mask,
    }


def analyze_groups(outlier_mask, group_size):
    N = len(outlier_mask)
    pad = (group_size - N % group_size) % group_size
    if pad > 0:
        outlier_mask = np.concatenate([outlier_mask, np.zeros(pad, dtype=bool)])
    groups = outlier_mask.reshape(-1, group_size)
    outliers_per_group = groups.sum(axis=1)
    num_groups = groups.shape[0]

    groups_with_outlier = (outliers_per_group >= 1).sum()
    groups_with_multi   = (outliers_per_group >= 2).sum()
    pct_with_outlier = groups_with_outlier / num_groups * 100
    pct_with_multi   = groups_with_multi / num_groups * 100

    if groups_with_outlier > 0:
        avg_outliers = outliers_per_group[outliers_per_group >= 1].mean()
    else:
        avg_outliers = 0

    victims_per_group = np.where(outliers_per_group >= 1, group_size - 1, 0)
    victim_waste_pct = victims_per_group.sum() / N * 100

    return {
        'pct_groups_with_outlier':      pct_with_outlier,
        'pct_groups_with_multi_outlier': pct_with_multi,
        'avg_outliers_per_active_group': avg_outliers,
        'victim_waste_pct':              victim_waste_pct,
    }


# ============================================================
# Main analysis
# ============================================================
print(f"\nAnalyzing {len(activations)} activation tensors...\n")

all_results = []
for name in sorted(activations.keys()):
    tensor = activations[name]
    print(f"  Analyzing: {name} shape={tuple(tensor.shape)}")
    stats = analyze_tensor(tensor, name, sigma_threshold=SIGMA_THRESHOLD)
    for gs in GROUP_SIZES:
        group_stats = analyze_groups(stats['outlier_mask'], gs)
        for k, v in group_stats.items():
            stats[f'g{gs}_{k}'] = v
    del stats['outlier_mask']
    all_results.append(stats)

# ============================================================
# Save CSV
# ============================================================
df = pd.DataFrame(all_results)
csv_path = f"{OUTPUT_DIR}/summary.csv"
df.to_csv(csv_path, index=False)
print(f"\n✓ Saved per-layer statistics to: {csv_path}")

# ============================================================
# Overall summary
# ============================================================
def write_summary():
    lines = []
    lines.append("=" * 70)
    lines.append(f"ACTIVATION OUTLIER ANALYSIS - {MODEL_NAME}")
    lines.append(f"Threshold: {SIGMA_THRESHOLD}σ")
    lines.append(f"Data: SST-2 validation, {NUM_SAMPLES} samples, max_len={MAX_LEN}")
    lines.append(f"Total layers analyzed: {len(all_results)}")
    lines.append("=" * 70)
    lines.append("")

    total_elements = df['n_elements'].sum()
    total_outliers = df['n_outliers'].sum()
    lines.append(f"Total activation elements: {total_elements:,}")
    lines.append(f"Total outliers:            {total_outliers:,}")
    lines.append(f"Overall outlier ratio:     {total_outliers/total_elements*100:.4f}%")
    lines.append("")

    lines.append("-" * 70)
    lines.append("PER-LAYER OUTLIER RATIO")
    lines.append("-" * 70)
    lines.append(f"Mean:   {df['outlier_ratio_pct'].mean():.4f}%")
    lines.append(f"Median: {df['outlier_ratio_pct'].median():.4f}%")
    lines.append(f"Min:    {df['outlier_ratio_pct'].min():.4f}%")
    lines.append(f"Max:    {df['outlier_ratio_pct'].max():.4f}%")
    lines.append("")

    lines.append("-" * 70)
    lines.append("MAX SIGMA")
    lines.append("-" * 70)
    lines.append(f"Mean:   {df['max_sigma'].mean():.2f}σ")
    lines.append(f"Median: {df['max_sigma'].median():.2f}σ")
    lines.append(f"Min:    {df['max_sigma'].min():.2f}σ")
    lines.append(f"Max:    {df['max_sigma'].max():.2f}σ  ← most extreme outlier")
    lines.append("")

    lines.append("-" * 70)
    lines.append("OUTLIER VALUE DISTRIBUTION (weighted by outlier count)")
    lines.append("-" * 70)
    weights = df['n_outliers'].values
    for col, label in [
        ('dist_3_6_pct',   '3σ - 6σ'),
        ('dist_6_10_pct',  '6σ - 10σ'),
        ('dist_10_50_pct', '10σ - 50σ'),
        ('dist_50p_pct',   '50σ+'),
    ]:
        avg = np.average(df[col].values, weights=weights) if weights.sum() > 0 else 0
        lines.append(f"  {label:12s} : {avg:6.2f}%  of all outliers")
    lines.append("")

    lines.append("=" * 70)
    lines.append("GROUP-WISE ANALYSIS (weighted by tensor size)")
    lines.append("=" * 70)
    for gs in GROUP_SIZES:
        lines.append(f"\n  Group size = {gs}:")
        for col_key, label in [
            (f'g{gs}_pct_groups_with_outlier',       '  Groups with outlier    '),
            (f'g{gs}_pct_groups_with_multi_outlier', '  Groups with 2+ outliers'),
            (f'g{gs}_victim_waste_pct',              '  Victim waste ratio     '),
        ]:
            avg = np.average(df[col_key].values, weights=df['n_elements'].values)
            lines.append(f"  {label} : {avg:7.4f}%")

    lines.append("")
    lines.append("=" * 70)
    lines.append("KEY QUESTIONS (ACTIVATION vs WEIGHT):")
    lines.append("=" * 70)
    lines.append("")
    p50 = np.average(df['dist_50p_pct'].values, weights=weights) if weights.sum() > 0 else 0
    p10 = np.average(df['dist_10_50_pct'].values, weights=weights) if weights.sum() > 0 else 0
    max_sig = df['max_sigma'].max()
    lines.append(f"Q1: Do activations have MORE extreme outliers than weights?")
    lines.append(f"    Max sigma observed: {max_sig:.1f}σ")
    lines.append(f"    Outliers > 50σ:     {p50:.2f}%")
    lines.append(f"    Outliers 10-50σ:    {p10:.2f}%")
    if max_sig > 100:
        lines.append(f"    → YES! activations have extreme outliers (paper's finding)")
        lines.append(f"    → More bits for outlier abfloat MIGHT actually help here")
    else:
        lines.append(f"    → Similar to weights, no extreme outliers")
    lines.append("")

    g2m = np.average(df['g2_pct_groups_with_multi_outlier'].values, weights=df['n_elements'].values)
    g4m = np.average(df['g4_pct_groups_with_multi_outlier'].values, weights=df['n_elements'].values)
    g8m = np.average(df['g8_pct_groups_with_multi_outlier'].values, weights=df['n_elements'].values)
    lines.append(f"Q2: Group collision rates:")
    lines.append(f"    Group=2 outlier-outlier collision: {g2m:.4f}%")
    lines.append(f"    Group=4 (2+ outliers per group):   {g4m:.4f}%")
    lines.append(f"    Group=8 (2+ outliers per group):   {g8m:.4f}%")
    lines.append("")

    lines.append(f"Q3: Victim waste comparison:")
    for gs in GROUP_SIZES:
        v = np.average(df[f'g{gs}_victim_waste_pct'].values, weights=df['n_elements'].values)
        lines.append(f"    Group={gs}: {v:.4f}% of activations become victims")
    lines.append("")

    return '\n'.join(lines)


summary_text = write_summary()
with open(f"{OUTPUT_DIR}/overall.txt", 'w') as f:
    f.write(summary_text)
print(f"✓ Saved overall summary to: {OUTPUT_DIR}/overall.txt")
print("\n" + summary_text)

# ============================================================
# Plots (same 4 as weight version)
# ============================================================
print("\nGenerating plots...")

# Plot 1: Outlier ratio per layer
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(df)), df['outlier_ratio_pct'].values)
ax.set_xlabel('Layer index')
ax.set_ylabel('Outlier ratio (%)')
ax.set_title(f'{MODEL_NAME} ACTIVATIONS: Outlier ratio per layer (>3σ)')
ax.axhline(y=df['outlier_ratio_pct'].mean(), color='r', linestyle='--',
           label=f'Mean = {df["outlier_ratio_pct"].mean():.3f}%')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/outlier_ratio_per_layer.png', dpi=100)
plt.close()

# Plot 2: Max sigma per layer
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(df)), df['max_sigma'].values, color='orange')
ax.set_xlabel('Layer index')
ax.set_ylabel('Max σ (most extreme activation outlier)')
ax.set_title(f'{MODEL_NAME} ACTIVATIONS: Max σ per layer')
ax.axhline(y=32, color='green', linestyle='--', label='OliVe abfloat threshold (32σ, 4-bit)')
ax.axhline(y=384, color='blue', linestyle='--', label='OliVe abfloat range max (384σ)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/max_sigma_per_layer.png', dpi=100)
plt.close()

# Plot 3: Outlier distribution
fig, ax = plt.subplots(figsize=(10, 6))
weights = df['n_outliers'].values
buckets = ['3-6σ', '6-10σ', '10-50σ', '50σ+']
values = [
    np.average(df['dist_3_6_pct'].values,   weights=weights) if weights.sum() > 0 else 0,
    np.average(df['dist_6_10_pct'].values,  weights=weights) if weights.sum() > 0 else 0,
    np.average(df['dist_10_50_pct'].values, weights=weights) if weights.sum() > 0 else 0,
    np.average(df['dist_50p_pct'].values,   weights=weights) if weights.sum() > 0 else 0,
]
ax.bar(buckets, values, color=['lightblue', 'skyblue', 'steelblue', 'darkblue'])
ax.set_ylabel('% of all activation outliers')
ax.set_title(f'{MODEL_NAME} ACTIVATIONS: How extreme are the outliers?')
for i, v in enumerate(values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/outlier_distribution.png', dpi=100)
plt.close()

# Plot 4: Group waste comparison
fig, ax = plt.subplots(figsize=(10, 6))
group_sizes = GROUP_SIZES
waste_pcts = [
    np.average(df[f'g{gs}_victim_waste_pct'].values, weights=df['n_elements'].values)
    for gs in group_sizes
]
active_pcts = [
    np.average(df[f'g{gs}_pct_groups_with_outlier'].values, weights=df['n_elements'].values)
    for gs in group_sizes
]
x = np.arange(len(group_sizes))
w = 0.35
b1 = ax.bar(x - w/2, active_pcts, w, label='% groups with outlier', color='coral')
b2 = ax.bar(x + w/2, waste_pcts,  w, label='% victim waste',        color='steelblue')
ax.set_xlabel('Group size')
ax.set_ylabel('%')
ax.set_title(f'{MODEL_NAME} ACTIVATIONS: Group activation vs Victim waste')
ax.set_xticks(x)
ax.set_xticklabels([f'g={g}' for g in group_sizes])
ax.legend()
for bars in [b1, b2]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.05, f'{h:.2f}%',
                ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/group_waste_comparison.png', dpi=100)
plt.close()

print(f"✓ Saved plots to: {OUTPUT_DIR}/plots/")
print("\nDone!")