import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

def data_prepare(data):

    results = torch.load(data, weights_only=False)
    vis_flows, attn_allocs = [], []

    for i, result in enumerate(results):
        vis_flow, attn_alloc = result
        vis_flows.append(vis_flow)
        attn_allocs.append(attn_alloc)

    vis_flows = np.array(vis_flows)
    attn_allocs = np.array(attn_allocs)

    vis_flows = vis_flows.mean(axis = 0).transpose(1, 0)
    attn_allocs = attn_allocs.mean(axis = 0).transpose(1, 0)

    return (vis_flows, attn_allocs)

def plot_vis_flow(data, path):
    # Filter out channels that are all zeros (across all layers)
    non_zero_mask = np.any(data > 1e-10, axis=1)
    filtered_data = data[non_zero_mask]
    filtered_indices = np.where(non_zero_mask)[0]
    
    original_channel = data.shape[0]
    channel = filtered_data.shape[0]
    
    if channel == 0:
        print("Warning: All channels are zero, cannot plot")
        return
    
    data_min = np.min(filtered_data)
    data_max = np.max(filtered_data)
    if data_max - data_min > 1e-9:
        filtered_data = (filtered_data - data_min) / (data_max - data_min)

    x = np.arange(filtered_data.shape[1])
    
    # Get labels and colors based on original channel count
    if original_channel == 7:
        all_colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
        all_labels = [
            'Intra-Visual Flow',
            'Text→Visual Flow',
            'Visual→Text Flow',
            'Audio→Visual Flow',
            'Visual→Audio Flow',
            'Text→Audio Flow',
            'Audio→Text Flow'
        ]
    elif original_channel == 4:
        all_colors = ['#bf1e2e', '#73bad6', '#f0c419', '#4B74B2']
        all_labels = ['Intra-Visual Flow', 'Visual→Text Flow', 'Visual→Audio Flow', 'Audio→Text Flow']
    elif original_channel == 2:
        all_colors = ['#bf1e2e', '#73bad6']
        all_labels = ['Intra-Visual Flow', 'Visual-Textual Flow']
    else:
        raise ValueError(f"Unsupported saliency channel count: {original_channel}")
    
    # Filter colors and labels to match filtered data
    colors = [all_colors[i] for i in filtered_indices]
    labels = [all_labels[i] for i in filtered_indices]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    
    bar_width = 0.8
    
    for idx in range(channel):
        ax.bar(
            x,
            filtered_data[idx],
            label=labels[idx],
            alpha=0.6,
            color=colors[idx],
            width=bar_width,
        )

    font_properties = {
        'family': 'serif',
        'weight': 'bold',
        'size': 16
    }

    plt.xlabel('Transformer Layer', fontdict=font_properties)
    plt.ylabel('Normalized Saliency', fontdict=font_properties)
    
    plt.legend(
        prop={'family': 'serif', 'weight': 'bold', 'size': 8},
        ncol=2,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.2,
        frameon=True,
        borderpad=0.3,
        loc='upper right')

    plt.xticks(fontsize=10, fontweight='bold', family='serif')
    plt.yticks(fontsize=10, fontweight='bold', family='serif')

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout(pad=1.2)
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()


def _attn_labels_and_colors(channel_count: int):
    if channel_count == 4:
        colors = ['#ff5e65', '#90bee0', '#f0c419', '#4B74B2']
        labels = ['System Prompts', 'Visual Tokens', 'Audio Tokens', 'User Instructions']
    elif channel_count == 3:
        colors = ['#ff5e65', '#90bee0', '#4B74B2']
        labels = ['System Prompts', 'Visual Features', 'User Instructions']
    else:
        raise ValueError(f"Unsupported channel count: {channel_count}")
    return labels, colors


def plot_attn_alloc(data, path):

    normalized_data = data / data.sum(axis=0, keepdims=True)
    num_layers = normalized_data.shape[1]
    x = np.arange(num_layers)

    labels, colors = _attn_labels_and_colors(normalized_data.shape[0])

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    bottom = np.zeros(num_layers)
    for idx, (segment, color, label) in enumerate(zip(normalized_data, colors, labels)):
        ax.bar(
            x,
            segment,
            bottom=bottom,
            color=color,
            edgecolor='black',
            linewidth=1,
            label=label,
        )
        bottom += segment

    ax.set_xlabel('Transformer Layer', fontsize=12, fontweight='bold', fontname='serif')
    ax.set_ylabel('Attention Allocation', fontsize=12, fontweight='bold', fontname='serif')
    ax.set_xticks(x[::3])
    ax.set_xlim(-1.5, num_layers - 0.5)

    ax.tick_params(axis='both', labelsize=15)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('serif')
        label.set_fontweight('bold')

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(labels), frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_fontsize(12)
        text.set_fontweight('bold')
        text.set_fontname('serif')

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()


def save_attn_alloc_json(data, path):
    normalized_data = data / data.sum(axis=0, keepdims=True)
    labels, _ = _attn_labels_and_colors(normalized_data.shape[0])
    num_layers = normalized_data.shape[1]

    records = []
    for layer_idx in range(num_layers):
        entry = {"layer": int(layer_idx)}
        for seg_idx, label in enumerate(labels):
            entry[label] = round(float(normalized_data[seg_idx, layer_idx] * 100.0), 4)
        records.append(entry)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _vis_flow_labels(channel_count: int):
    if channel_count == 7:
        return [
            'Intra-Visual Flow',
            'Text→Visual Flow',
            'Visual→Text Flow',
            'Audio→Visual Flow',
            'Visual→Audio Flow',
            'Text→Audio Flow',
            'Audio→Text Flow'
        ]
    elif channel_count == 4:
        return ['Intra-Visual Flow', 'Visual→Text Flow', 'Visual→Audio Flow', 'Audio→Text Flow']
    elif channel_count == 2:
        return ['Intra-Visual Flow', 'Visual-Textual Flow']
    else:
        raise ValueError(f"Unsupported saliency channel count: {channel_count}")


def save_vis_flow_json(data, path):
    """
    Save vis_flow data to JSON.
    Note: data is the original averaged values before normalization.
    """
    channel = data.shape[0]
    num_layers = data.shape[1]
    labels = _vis_flow_labels(channel)

    records = []
    for layer_idx in range(num_layers):
        entry = {"layer": int(layer_idx)}
        for ch_idx, label in enumerate(labels):
            value = float(data[ch_idx, layer_idx])
            if value == 0.0:
                entry[label] = 0.0
            elif abs(value) < 1e-6:
                entry[label] = float(f"{value:.6e}")
            else:
                entry[label] = round(value, 6)
        records.append(entry)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot visualization flow and attention allocation from vis_flow results")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pt file (merged_results.pt)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots (default: same as input file directory)")
    parser.add_argument("--output-prefix", type=str, default="", help="Prefix for output filenames (default: empty)")
    parser.add_argument("--attn-json", type=str, default=None, help="Output path for per-layer attention percentages (JSON)")
    parser.add_argument("--vis-flow-json", type=str, default=None, help="Output path for per-layer visual flow values (JSON)")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input))
    os.makedirs(args.output_dir, exist_ok=True)
    
    vis_flows, attn_allocs = data_prepare(args.input)
    
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    attn_output = os.path.join(args.output_dir, f"{prefix}attn_allocs.png")
    vis_flow_output = os.path.join(args.output_dir, f"{prefix}vis_flows.png")
    attn_json_output = args.attn_json or os.path.join(args.output_dir, f"{prefix}attn_allocs.json")
    vis_flow_json_output = args.vis_flow_json or os.path.join(args.output_dir, f"{prefix}vis_flows.json")
    
    plot_attn_alloc(attn_allocs, attn_output)
    save_attn_alloc_json(attn_allocs, attn_json_output)
    
    vis_flows_original = vis_flows.copy()
    plot_vis_flow(vis_flows, vis_flow_output)
    save_vis_flow_json(vis_flows_original, vis_flow_json_output)
