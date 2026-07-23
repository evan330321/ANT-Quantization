"""
Outlier Analysis Script for BERT-base
=====================================
Analyzes weight tensors of BERT-base to understand outlier distribution
and how it interacts with different OVP group sizes (2, 4, 8).

Usage:
    python analyze_outliers.py
    
Output:
    - outlier_analysis/summary.csv     : per-layer statistics
    - outlier_analysis/overall.txt     : aggregate statistics
    - outlier_analysis/plots/*.png     : visualizations
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModel
from collections import defaultdict

# ============================================================
# Config
# ============================================================
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./outlier_analysis"
SIGMA_THRESHOLD = 3.0  # OliVe paper uses 3-sigma rule
GROUP_SIZES = [2, 4, 8]

# ============================================================
# Setup
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

print(f"Loading model: {MODEL_NAME}")
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ============================================================
# Analysis functions
# ============================================================
def analyze_tensor(tensor, name, sigma_threshold=3.0):
    """Analyze a single weight tensor for outlier statistics."""
    flat = tensor.flatten().float().cpu().numpy()
    
    mean = flat.mean()
    std = flat.std()
    abs_max = np.abs(flat).max()
    max_sigma = abs_max / std if std > 0 else 0
    
    # Outlier mask (values > 3-sigma)
    threshold = sigma_threshold * std
    outlier_mask = np.abs(flat - mean) > threshold
    n_outliers = outlier_mask.sum()
    outlier_ratio = n_outliers / len(flat)
    
    # Outlier value distribution (in units of sigma)
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
        'outlier_mask': outlier_mask,  # 保留給 group analysis
    }


def analyze_groups(outlier_mask, group_size):
    """
    Given a boolean outlier mask (flat tensor), analyze grouping statistics.
    
    Returns:
        - pct_groups_with_outlier: % of groups that contain at least one outlier
        - pct_groups_with_multi_outlier: % of groups with 2+ outliers (collision)
        - avg_outliers_per_group_with_outlier: average outliers count in outlier-containing groups
        - victim_waste_pct: total normal values sacrificed as victims (% of tensor)
    """
    N = len(outlier_mask)
    # Pad to multiple of group_size
    pad = (group_size - N % group_size) % group_size
    if pad > 0:
        outlier_mask = np.concatenate([outlier_mask, np.zeros(pad, dtype=bool)])
    
    # Reshape into groups
    groups = outlier_mask.reshape(-1, group_size)
    num_groups = groups.shape[0]
    
    # Count outliers per group
    outliers_per_group = groups.sum(axis=1)
    
    groups_with_outlier = (outliers_per_group >= 1).sum()
    groups_with_multi   = (outliers_per_group >= 2).sum()
    
    pct_with_outlier = groups_with_outlier / num_groups * 100
    pct_with_multi   = groups_with_multi / num_groups * 100
    
    if groups_with_outlier > 0:
        avg_outliers = outliers_per_group[outliers_per_group >= 1].mean()
    else:
        avg_outliers = 0
    
    # Victim waste: for each group with outlier, (group_size - n_outliers) values become victims
    # 假設每個 group 最多留 1 個 outlier，其他都是 victim
    victims_per_group = np.where(
        outliers_per_group >= 1,
        group_size - 1,  # 只留 1 個 outlier，其他都變 victim
        0
    )
    total_victims = victims_per_group.sum()
    victim_waste_pct = total_victims / N * 100
    
    return {
        'pct_groups_with_outlier':      pct_with_outlier,
        'pct_groups_with_multi_outlier': pct_with_multi,
        'avg_outliers_per_active_group': avg_outliers,
        'victim_waste_pct':              victim_waste_pct,
    }


# ============================================================
# Main analysis loop
# ============================================================
print(f"\nAnalyzing weight tensors (threshold = {SIGMA_THRESHOLD}σ)...\n")

all_results = []
skipped = 0

for name, param in model.named_parameters():
    # Only analyze weight matrices (skip biases, layer norms, embeddings for now)
    if 'weight' not in name:
        continue
    # Skip LayerNorm weights (they're 1-D scaling factors, not interesting)
    if 'LayerNorm' in name or 'layer_norm' in name:
        skipped += 1
        continue
    # Skip embeddings for now (embedding weights are different)
    if 'embedding' in name.lower():
        skipped += 1
        continue
    
    print(f"  Analyzing: {name} shape={tuple(param.shape)}")
    stats = analyze_tensor(param.data, name, sigma_threshold=SIGMA_THRESHOLD)
    
    # Add group-wise statistics
    for gs in GROUP_SIZES:
        group_stats = analyze_groups(stats['outlier_mask'], gs)
        for k, v in group_stats.items():
            stats[f'g{gs}_{k}'] = v
    
    # Remove the raw mask before saving (too big)
    del stats['outlier_mask']
    all_results.append(stats)

print(f"\nAnalyzed {len(all_results)} weight tensors (skipped {skipped}).\n")

# ============================================================
# Save per-layer CSV
# ============================================================
df = pd.DataFrame(all_results)
csv_path = f"{OUTPUT_DIR}/summary.csv"
df.to_csv(csv_path, index=False)
print(f"✓ Saved per-layer statistics to: {csv_path}")

# ============================================================
# Overall summary
# ============================================================
def write_summary():
    lines = []
    lines.append("=" * 70)
    lines.append(f"OUTLIER ANALYSIS SUMMARY - {MODEL_NAME}")
    lines.append(f"Threshold: {SIGMA_THRESHOLD}σ (3-sigma rule)")
    lines.append(f"Total layers analyzed: {len(all_results)}")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall outlier stats
    total_elements = df['n_elements'].sum()
    total_outliers = df['n_outliers'].sum()
    lines.append(f"Total weight elements: {total_elements:,}")
    lines.append(f"Total outliers:        {total_outliers:,}")
    lines.append(f"Overall outlier ratio: {total_outliers/total_elements*100:.4f}%")
    lines.append("")
    
    # Per-layer outlier ratio distribution
    lines.append("-" * 70)
    lines.append("PER-LAYER OUTLIER RATIO")
    lines.append("-" * 70)
    lines.append(f"Mean:   {df['outlier_ratio_pct'].mean():.4f}%")
    lines.append(f"Median: {df['outlier_ratio_pct'].median():.4f}%")
    lines.append(f"Min:    {df['outlier_ratio_pct'].min():.4f}%")
    lines.append(f"Max:    {df['outlier_ratio_pct'].max():.4f}%")
    lines.append("")
    
    # Max sigma distribution
    lines.append("-" * 70)
    lines.append("MAX SIGMA (how extreme is the biggest value?)")
    lines.append("-" * 70)
    lines.append(f"Mean:   {df['max_sigma'].mean():.2f}σ")
    lines.append(f"Median: {df['max_sigma'].median():.2f}σ")
    lines.append(f"Min:    {df['max_sigma'].min():.2f}σ")
    lines.append(f"Max:    {df['max_sigma'].max():.2f}σ  ← 最極端的 outlier")
    lines.append("")
    
    # Outlier value distribution (aggregate)
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
    
    # Group-wise stats
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
    lines.append("KEY QUESTIONS TO CHECK:")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Q1: Are outliers extreme? (>50σ)")
    p50 = np.average(df['dist_50p_pct'].values, weights=weights) if weights.sum() > 0 else 0
    if p50 > 5:
        lines.append(f"    YES - {p50:.1f}% of outliers are >50σ → high precision helps!")
    else:
        lines.append(f"    NO  - only {p50:.1f}% of outliers are >50σ → 4-bit may be enough")
    lines.append("")
    
    lines.append("Q2: Does larger group help catch more outliers?")
    g2_mult = np.average(df['g2_pct_groups_with_multi_outlier'].values, weights=df['n_elements'].values)
    g4_mult = np.average(df['g4_pct_groups_with_multi_outlier'].values, weights=df['n_elements'].values)
    g8_mult = np.average(df['g8_pct_groups_with_multi_outlier'].values, weights=df['n_elements'].values)
    lines.append(f"    Group=2 outlier-outlier collision: {g2_mult:.4f}%")
    lines.append(f"    Group=4 (2+ outliers per group):   {g4_mult:.4f}%")
    lines.append(f"    Group=8 (2+ outliers per group):   {g8_mult:.4f}%")
    lines.append("")
    
    lines.append("Q3: How much victim waste do we pay?")
    for gs in GROUP_SIZES:
        v = np.average(df[f'g{gs}_victim_waste_pct'].values, weights=df['n_elements'].values)
        lines.append(f"    Group={gs}: {v:.4f}% of all values become victims")
    lines.append("")
    
    return '\n'.join(lines)


summary_text = write_summary()
summary_path = f"{OUTPUT_DIR}/overall.txt"
with open(summary_path, 'w') as f:
    f.write(summary_text)
print(f"✓ Saved overall summary to: {summary_path}")

print("\n" + summary_text)

# ============================================================
# Plots
# ============================================================
print("\nGenerating plots...")

# Plot 1: Outlier ratio per layer
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(len(df)), df['outlier_ratio_pct'].values)
ax.set_xlabel('Layer index')
ax.set_ylabel('Outlier ratio (%)')
ax.set_title(f'{MODEL_NAME}: Outlier ratio per weight tensor (>3σ)')
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
ax.set_ylabel('Max σ (most extreme outlier)')
ax.set_title(f'{MODEL_NAME}: Max σ per weight tensor')
ax.axhline(y=32, color='green', linestyle='--', label='OliVe abfloat max (32σ, 4-bit)')
ax.axhline(y=384, color='blue', linestyle='--', label='OliVe abfloat range max (384σ)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/max_sigma_per_layer.png', dpi=100)
plt.close()

# Plot 3: Outlier distribution buckets
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
ax.set_ylabel('% of all outliers')
ax.set_title(f'{MODEL_NAME}: How extreme are the outliers?')
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
ax.set_title(f'{MODEL_NAME}: Group activation vs Victim waste')
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
print("\nDone! Check the output directory for full results.")