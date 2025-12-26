#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the ratio between multimodal tokens (visual + audio) and text tokens (system + user instructions).
"""

import argparse
import json
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_attn_allocs(json_path):
    """Load attention allocation JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def calculate_modality_ratios(data):
    """
    Calculate ratios between multimodal tokens and text tokens.
    
    Returns:
        dict with layer-wise ratios and statistics
    """
    layers = []
    multimodal_ratios = []  # (Visual + Audio) / (System + User)
    multimodal_sums = []
    text_sums = []
    
    for layer_data in data:
        layer_idx = layer_data['layer']
        visual = layer_data['Visual Tokens']
        audio = layer_data['Audio Tokens']
        system = layer_data['System Prompts']
        user = layer_data['User Instructions']
        
        multimodal_sum = visual + audio
        text_sum = system + user
        
        # Calculate ratio
        if text_sum > 0:
            ratio = multimodal_sum / text_sum
        else:
            ratio = float('inf')
        
        layers.append(layer_idx)
        multimodal_ratios.append(ratio)
        multimodal_sums.append(multimodal_sum)
        text_sums.append(text_sum)
    
    layers = np.array(layers)
    multimodal_ratios = np.array(multimodal_ratios)
    multimodal_sums = np.array(multimodal_sums)
    text_sums = np.array(text_sums)
    
    # Calculate statistics
    results = {
        'layers': layers.tolist(),
        'multimodal_ratios': multimodal_ratios.tolist(),
        'multimodal_sums': multimodal_sums.tolist(),
        'text_sums': text_sums.tolist(),
        'statistics': {
            'mean_ratio': float(np.mean(multimodal_ratios)),
            'std_ratio': float(np.std(multimodal_ratios)),
            'min_ratio': float(np.min(multimodal_ratios)),
            'max_ratio': float(np.max(multimodal_ratios)),
            'mean_multimodal': float(np.mean(multimodal_sums)),
            'mean_text': float(np.mean(text_sums)),
        }
    }
    
    # Early, Middle, Late analysis
    n = len(layers)
    early_end = n // 3
    late_start = 2 * n // 3
    
    results['early'] = {
        'mean_ratio': float(np.mean(multimodal_ratios[:early_end])),
        'mean_multimodal': float(np.mean(multimodal_sums[:early_end])),
        'mean_text': float(np.mean(text_sums[:early_end])),
    }
    
    results['middle'] = {
        'mean_ratio': float(np.mean(multimodal_ratios[early_end:late_start])),
        'mean_multimodal': float(np.mean(multimodal_sums[early_end:late_start])),
        'mean_text': float(np.mean(text_sums[early_end:late_start])),
    }
    
    results['late'] = {
        'mean_ratio': float(np.mean(multimodal_ratios[late_start:])),
        'mean_multimodal': float(np.mean(multimodal_sums[late_start:])),
        'mean_text': float(np.mean(text_sums[late_start:])),
    }
    
    # Trend analysis (correlation with layer index)
    if np.std(multimodal_ratios) > 1e-10:
        corr = np.corrcoef(layers, multimodal_ratios)[0, 1]
        results['trend'] = {
            'correlation': float(corr),
            'direction': 'increasing' if corr > 0.3 else 'decreasing' if corr < -0.3 else 'stable'
        }
    else:
        results['trend'] = {
            'correlation': 0.0,
            'direction': 'stable'
        }
    
    return results


def print_analysis(results):
    """Print analysis results."""
    print("="*80)
    print("MULTIMODAL vs TEXT TOKENS RATIO ANALYSIS")
    print("="*80)
    
    stats = results['statistics']
    print(f"\nOverall Statistics:")
    print(f"  Multimodal (Visual + Audio) average: {stats['mean_multimodal']:.2f}%")
    print(f"  Text (System + User) average: {stats['mean_text']:.2f}%")
    print(f"  Ratio average: {stats['mean_ratio']:.4f}")
    print(f"  Ratio std: {stats['std_ratio']:.4f}")
    print(f"  Ratio min: {stats['min_ratio']:.4f} (Layer {np.argmin(results['multimodal_ratios'])})")
    print(f"  Ratio max: {stats['max_ratio']:.4f} (Layer {np.argmax(results['multimodal_ratios'])})")
    
    print(f"\nLayer Group Analysis:")
    print(f"  Early Layers (0-{len(results['layers'])//3-1}):")
    print(f"    Ratio: {results['early']['mean_ratio']:.4f}")
    print(f"    Multimodal: {results['early']['mean_multimodal']:.2f}%, Text: {results['early']['mean_text']:.2f}%")
    
    mid_start = len(results['layers'])//3
    mid_end = 2*len(results['layers'])//3-1
    print(f"  Middle Layers ({mid_start}-{mid_end}):")
    print(f"    Ratio: {results['middle']['mean_ratio']:.4f}")
    print(f"    Multimodal: {results['middle']['mean_multimodal']:.2f}%, Text: {results['middle']['mean_text']:.2f}%")
    
    late_start = 2*len(results['layers'])//3
    print(f"  Late Layers ({late_start}-{results['layers'][-1]}):")
    print(f"    Ratio: {results['late']['mean_ratio']:.4f}")
    print(f"    Multimodal: {results['late']['mean_multimodal']:.2f}%, Text: {results['late']['mean_text']:.2f}%")
    
    print(f"\nTrend Analysis:")
    trend = results['trend']
    print(f"  Direction: {trend['direction']} (correlation={trend['correlation']:.3f})")
    
    early_late_ratio = results['late']['mean_ratio'] / results['early']['mean_ratio'] if results['early']['mean_ratio'] > 0 else float('inf')
    if early_late_ratio != float('inf'):
        print(f"  Early vs Late ratio change: {early_late_ratio:.2f}x")
        if early_late_ratio > 1.0:
            print(f"    → Multimodal attention increased {early_late_ratio:.2f}x in Late layers")
        elif early_late_ratio < 1.0:
            print(f"    → Multimodal attention decreased {1.0/early_late_ratio:.2f}x in Late layers")
    
    print(f"\nLayer-by-Layer Detailed Ratios:")
    print(f"  {'Layer':<8} {'Multimodal%':<15} {'Text%':<15} {'Ratio':<10}")
    print("-" * 60)
    for i, layer in enumerate(results['layers']):
        print(f"  {layer:<8} {results['multimodal_sums'][i]:<15.2f} {results['text_sums'][i]:<15.2f} {results['multimodal_ratios'][i]:<10.4f}")


def plot_ratios(results, output_path):
    """Plot the ratio across layers."""
    layers = np.array(results['layers'])
    ratios = np.array(results['multimodal_ratios'])
    multimodal_sums = np.array(results['multimodal_sums'])
    text_sums = np.array(results['text_sums'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Ratio
    ax1.plot(layers, ratios, 'o-', linewidth=2, markersize=6, color='#2ca02c', label='Multimodal/Text Ratio')
    ax1.axhline(y=results['statistics']['mean_ratio'], color='r', linestyle='--', linewidth=1.5, label=f"Mean: {results['statistics']['mean_ratio']:.4f}")
    ax1.set_xlabel('Transformer Layer', fontsize=12, fontweight='bold', fontname='serif')
    ax1.set_ylabel('Ratio (Multimodal / Text)', fontsize=12, fontweight='bold', fontname='serif')
    ax1.set_title('Multimodal vs Text Tokens Ratio Across Layers', fontsize=14, fontweight='bold', fontname='serif')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(layers[::3])
    
    # Plot 2: Absolute percentages
    ax2.plot(layers, multimodal_sums, 'o-', linewidth=2, markersize=6, color='#1f77b4', label='Multimodal (Visual + Audio)')
    ax2.plot(layers, text_sums, 's-', linewidth=2, markersize=6, color='#ff7f0e', label='Text (System + User)')
    ax2.set_xlabel('Transformer Layer', fontsize=12, fontweight='bold', fontname='serif')
    ax2.set_ylabel('Attention Allocation (%)', fontsize=12, fontweight='bold', fontname='serif')
    ax2.set_title('Absolute Attention Allocation', fontsize=14, fontweight='bold', fontname='serif')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(layers[::3])
    
    for ax in [ax1, ax2]:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('serif')
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multimodal vs text tokens ratio")
    parser.add_argument("--input", type=str, required=True, help="Path to attn_allocs.json file")
    parser.add_argument("--output-json", type=str, default=None, help="Output JSON file path (optional)")
    parser.add_argument("--output-plot", type=str, default=None, help="Output plot file path (optional)")
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.input}")
    data = load_attn_allocs(args.input)
    print(f"Loaded {len(data)} layers")
    
    results = calculate_modality_ratios(data)
    print_analysis(results)
    
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_json}")
    
    if args.output_plot:
        if HAS_MATPLOTLIB:
            plot_ratios(results, args.output_plot)
        else:
            print(f"\nWarning: matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()

