"""
Analyze directional differences in saliency values.

This script compares bidirectional saliency flows (e.g., Visual→Text vs Text→Visual)
to understand asymmetry in attention patterns.
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_vis_flow_json(json_path):
    """Load vis_flow JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_directional_differences(data):
    """
    Analyze differences between bidirectional saliency flows.
    
    Returns:
        dict with comparison results
    """
    num_layers = len(data)
    
    # Extract values for each direction
    vis_txt = []  # Visual→Text
    txt_vis = []  # Text→Visual
    vis_aud = []  # Visual→Audio
    aud_vis = []  # Audio→Visual
    txt_aud = []  # Text→Audio
    aud_txt = []  # Audio→Text
    
    for layer_data in data:
        layer_idx = layer_data['layer']
        
        # Check which channels are available
        if 'Visual→Text Flow' in layer_data and 'Text→Visual Flow' in layer_data:
            vis_txt_val = float(layer_data['Visual→Text Flow']) if isinstance(layer_data['Visual→Text Flow'], str) else layer_data['Visual→Text Flow']
            txt_vis_val = float(layer_data['Text→Visual Flow']) if isinstance(layer_data['Text→Visual Flow'], str) else layer_data['Text→Visual Flow']
            vis_txt.append(vis_txt_val)
            txt_vis.append(txt_vis_val)
        
        if 'Visual→Audio Flow' in layer_data and 'Audio→Visual Flow' in layer_data:
            vis_aud_val = float(layer_data['Visual→Audio Flow']) if isinstance(layer_data['Visual→Audio Flow'], str) else layer_data['Visual→Audio Flow']
            aud_vis_val = float(layer_data['Audio→Visual Flow']) if isinstance(layer_data['Audio→Visual Flow'], str) else layer_data['Audio→Visual Flow']
            vis_aud.append(vis_aud_val)
            aud_vis.append(aud_vis_val)
        
        if 'Text→Audio Flow' in layer_data and 'Audio→Text Flow' in layer_data:
            txt_aud_val = float(layer_data['Text→Audio Flow']) if isinstance(layer_data['Text→Audio Flow'], str) else layer_data['Text→Audio Flow']
            aud_txt_val = float(layer_data['Audio→Text Flow']) if isinstance(layer_data['Audio→Text Flow'], str) else layer_data['Audio→Text Flow']
            txt_aud.append(txt_aud_val)
            aud_txt.append(aud_txt_val)
    
    results = {}
    
    # Visual - Text comparison
    if vis_txt and txt_vis:
        vis_txt_arr = np.array(vis_txt)
        txt_vis_arr = np.array(txt_vis)
        diff = vis_txt_arr - txt_vis_arr
        ratio = np.where((txt_vis_arr != 0) & (vis_txt_arr != 0), 
                        vis_txt_arr / txt_vis_arr, 
                        np.nan)
        
        results['Visual_Text'] = {
            'Visual→Text': {
                'mean': float(np.mean(vis_txt_arr)),
                'std': float(np.std(vis_txt_arr)),
                'max': float(np.max(vis_txt_arr)),
                'min': float(np.min(vis_txt_arr)),
            },
            'Text→Visual': {
                'mean': float(np.mean(txt_vis_arr)),
                'std': float(np.std(txt_vis_arr)),
                'max': float(np.max(txt_vis_arr)),
                'min': float(np.min(txt_vis_arr)),
            },
            'difference': {
                'mean': float(np.mean(diff)),
                'std': float(np.std(diff)),
                'max': float(np.max(diff)),
                'min': float(np.min(diff)),
            },
            'ratio': {
                'mean': float(np.nanmean(ratio)),
                'median': float(np.nanmedian(ratio)),
                'std': float(np.nanstd(ratio)),
            }
        }
    
    # Visual - Audio comparison
    if vis_aud and aud_vis:
        vis_aud_arr = np.array(vis_aud)
        aud_vis_arr = np.array(aud_vis)
        diff = vis_aud_arr - aud_vis_arr
        ratio = np.where((aud_vis_arr != 0) & (vis_aud_arr != 0), 
                        vis_aud_arr / aud_vis_arr, 
                        np.nan)
        
        results['Visual_Audio'] = {
            'Visual→Audio': {
                'mean': float(np.mean(vis_aud_arr)),
                'std': float(np.std(vis_aud_arr)),
                'max': float(np.max(vis_aud_arr)),
                'min': float(np.min(vis_aud_arr)),
            },
            'Audio→Visual': {
                'mean': float(np.mean(aud_vis_arr)),
                'std': float(np.std(aud_vis_arr)),
                'max': float(np.max(aud_vis_arr)),
                'min': float(np.min(aud_vis_arr)),
            },
            'difference': {
                'mean': float(np.mean(diff)),
                'std': float(np.std(diff)),
                'max': float(np.max(diff)),
                'min': float(np.min(diff)),
            },
            'ratio': {
                'mean': float(np.nanmean(ratio)),
                'median': float(np.nanmedian(ratio)),
                'std': float(np.nanstd(ratio)),
            }
        }
    
    # Text - Audio comparison
    if txt_aud and aud_txt:
        txt_aud_arr = np.array(txt_aud)
        aud_txt_arr = np.array(aud_txt)
        diff = txt_aud_arr - aud_txt_arr
        ratio = np.where((aud_txt_arr != 0) & (txt_aud_arr != 0), 
                        txt_aud_arr / aud_txt_arr, 
                        np.nan)
        
        results['Text_Audio'] = {
            'Text→Audio': {
                'mean': float(np.mean(txt_aud_arr)),
                'std': float(np.std(txt_aud_arr)),
                'max': float(np.max(txt_aud_arr)),
                'min': float(np.min(txt_aud_arr)),
            },
            'Audio→Text': {
                'mean': float(np.mean(aud_txt_arr)),
                'std': float(np.std(aud_txt_arr)),
                'max': float(np.max(aud_txt_arr)),
                'min': float(np.min(aud_txt_arr)),
            },
            'difference': {
                'mean': float(np.mean(diff)),
                'std': float(np.std(diff)),
                'max': float(np.max(diff)),
                'min': float(np.min(diff)),
            },
            'ratio': {
                'mean': float(np.nanmean(ratio)),
                'median': float(np.nanmedian(ratio)),
                'std': float(np.nanstd(ratio)),
            }
        }
    
    return results


def analyze_layer_wise_trends(data):
    """
    Analyze how directional saliency changes across layers.
    
    Returns:
        dict with layer-wise trends
    """
    num_layers = len(data)
    
    # Extract layer-wise values
    trends = {
        'Visual_Text': {
            'Visual→Text': [],
            'Text→Visual': [],
            'layers': []
        },
        'Visual_Audio': {
            'Visual→Audio': [],
            'Audio→Visual': [],
            'layers': []
        },
        'Text_Audio': {
            'Text→Audio': [],
            'Audio→Text': [],
            'layers': []
        }
    }
    
    for layer_data in data:
        layer_idx = layer_data['layer']
        
        # Visual - Text
        if 'Visual→Text Flow' in layer_data and 'Text→Visual Flow' in layer_data:
            vis_txt_val = float(layer_data['Visual→Text Flow']) if isinstance(layer_data['Visual→Text Flow'], str) else layer_data['Visual→Text Flow']
            txt_vis_val = float(layer_data['Text→Visual Flow']) if isinstance(layer_data['Text→Visual Flow'], str) else layer_data['Text→Visual Flow']
            trends['Visual_Text']['Visual→Text'].append(vis_txt_val)
            trends['Visual_Text']['Text→Visual'].append(txt_vis_val)
            trends['Visual_Text']['layers'].append(layer_idx)
        
        # Visual - Audio
        if 'Visual→Audio Flow' in layer_data and 'Audio→Visual Flow' in layer_data:
            vis_aud_val = float(layer_data['Visual→Audio Flow']) if isinstance(layer_data['Visual→Audio Flow'], str) else layer_data['Visual→Audio Flow']
            aud_vis_val = float(layer_data['Audio→Visual Flow']) if isinstance(layer_data['Audio→Visual Flow'], str) else layer_data['Audio→Visual Flow']
            trends['Visual_Audio']['Visual→Audio'].append(vis_aud_val)
            trends['Visual_Audio']['Audio→Visual'].append(aud_vis_val)
            trends['Visual_Audio']['layers'].append(layer_idx)
        
        # Text - Audio
        if 'Text→Audio Flow' in layer_data and 'Audio→Text Flow' in layer_data:
            txt_aud_val = float(layer_data['Text→Audio Flow']) if isinstance(layer_data['Text→Audio Flow'], str) else layer_data['Text→Audio Flow']
            aud_txt_val = float(layer_data['Audio→Text Flow']) if isinstance(layer_data['Audio→Text Flow'], str) else layer_data['Audio→Text Flow']
            trends['Text_Audio']['Text→Audio'].append(txt_aud_val)
            trends['Text_Audio']['Audio→Text'].append(aud_txt_val)
            trends['Text_Audio']['layers'].append(layer_idx)
    
    # Calculate trends (early vs late layers)
    for pair_name, pair_trends in trends.items():
        if len(pair_trends['layers']) == 0:
            continue
        
        # Split into early (first 1/3), middle (middle 1/3), late (last 1/3)
        n = len(pair_trends['layers'])
        early_end = n // 3
        late_start = 2 * n // 3
        
        dir1_name = list(pair_trends.keys())[0]
        dir2_name = list(pair_trends.keys())[1]
        
        dir1_vals = np.array(pair_trends[dir1_name])
        dir2_vals = np.array(pair_trends[dir2_name])
        
        pair_trends['early'] = {
            dir1_name: {
                'mean': float(np.mean(dir1_vals[:early_end])),
                'max': float(np.max(dir1_vals[:early_end]))
            },
            dir2_name: {
                'mean': float(np.mean(dir2_vals[:early_end])),
                'max': float(np.max(dir2_vals[:early_end]))
            }
        }
        
        pair_trends['middle'] = {
            dir1_name: {
                'mean': float(np.mean(dir1_vals[early_end:late_start])),
                'max': float(np.max(dir1_vals[early_end:late_start]))
            },
            dir2_name: {
                'mean': float(np.mean(dir2_vals[early_end:late_start])),
                'max': float(np.max(dir2_vals[early_end:late_start]))
            }
        }
        
        pair_trends['late'] = {
            dir1_name: {
                'mean': float(np.mean(dir1_vals[late_start:])),
                'max': float(np.max(dir1_vals[late_start:]))
            },
            dir2_name: {
                'mean': float(np.mean(dir2_vals[late_start:])),
                'max': float(np.max(dir2_vals[late_start:]))
            }
        }
        
        # Calculate correlation with layer index (trend direction)
        layers_arr = np.array(pair_trends['layers'])
        if np.std(dir1_vals) > 1e-12:
            dir1_corr = np.corrcoef(layers_arr, dir1_vals)[0, 1]
            pair_trends['correlation'] = {dir1_name: float(dir1_corr) if not np.isnan(dir1_corr) else 0.0}
        else:
            pair_trends['correlation'] = {dir1_name: 0.0}
        
        if np.std(dir2_vals) > 1e-12:
            dir2_corr = np.corrcoef(layers_arr, dir2_vals)[0, 1]
            pair_trends['correlation'][dir2_name] = float(dir2_corr) if not np.isnan(dir2_corr) else 0.0
        else:
            pair_trends['correlation'][dir2_name] = 0.0
    
    return trends


def print_results(results, trends=None):
    """Print analysis results in a readable format."""
    print("\n" + "="*80)
    print("DIRECTIONAL SALIENCY ANALYSIS")
    print("="*80)
    
    for pair_name, pair_data in results.items():
        print(f"\n{pair_name.replace('_', ' - ')}:")
        print("-" * 80)
        
        # Get direction names
        dir1_name = list(pair_data.keys())[0]
        dir2_name = list(pair_data.keys())[1]
        
        print(f"\n{dir1_name}:")
        stats = pair_data[dir1_name]
        print(f"  Mean: {stats['mean']:.6e}")
        print(f"  Std:  {stats['std']:.6e}")
        print(f"  Min:  {stats['min']:.6e}")
        print(f"  Max:  {stats['max']:.6e}")
        
        print(f"\n{dir2_name}:")
        stats = pair_data[dir2_name]
        print(f"  Mean: {stats['mean']:.6e}")
        print(f"  Std:  {stats['std']:.6e}")
        print(f"  Min:  {stats['min']:.6e}")
        print(f"  Max:  {stats['max']:.6e}")
        
        print(f"\nDifference ({dir1_name} - {dir2_name}):")
        diff_stats = pair_data['difference']
        print(f"  Mean: {diff_stats['mean']:.6e}")
        print(f"  Std:  {diff_stats['std']:.6e}")
        print(f"  Min:  {diff_stats['min']:.6e}")
        print(f"  Max:  {diff_stats['max']:.6e}")
        
        print(f"\nRatio ({dir1_name} / {dir2_name}):")
        ratio_stats = pair_data['ratio']
        if not np.isnan(ratio_stats['mean']):
            print(f"  Mean:   {ratio_stats['mean']:.4f}x")
            print(f"  Median: {ratio_stats['median']:.4f}x")
            print(f"  Std:    {ratio_stats['std']:.4f}")
            if ratio_stats['mean'] > 1.0:
                print(f"  → {dir1_name} is {ratio_stats['mean']:.2f}x larger on average")
            elif ratio_stats['mean'] < 1.0:
                print(f"  → {dir2_name} is {1.0/ratio_stats['mean']:.2f}x larger on average")
            else:
                print(f"  → Values are approximately equal")
        else:
            print("  Cannot compute ratio (too many zeros)")
    
    # Print layer-wise trends
    if trends:
        print("\n" + "="*80)
        print("LAYER-WISE TRENDS")
        print("="*80)
        
        for pair_name, pair_trends in trends.items():
            if len(pair_trends.get('layers', [])) == 0:
                continue
            
            print(f"\n{pair_name.replace('_', ' - ')}:")
            print("-" * 80)
            
            dir1_name = list(pair_trends.keys())[0]
            dir2_name = list(pair_trends.keys())[1]
            
            # Early layers
            print(f"\nEarly Layers (0-{pair_trends['layers'][len(pair_trends['layers'])//3-1]}):")
            early = pair_trends['early']
            print(f"  {dir1_name}: mean={early[dir1_name]['mean']:.6e}, max={early[dir1_name]['max']:.6e}")
            print(f"  {dir2_name}: mean={early[dir2_name]['mean']:.6e}, max={early[dir2_name]['max']:.6e}")
            
            # Middle layers
            mid_start = len(pair_trends['layers'])//3
            mid_end = 2*len(pair_trends['layers'])//3-1
            print(f"\nMiddle Layers ({pair_trends['layers'][mid_start]}-{pair_trends['layers'][mid_end]}):")
            middle = pair_trends['middle']
            print(f"  {dir1_name}: mean={middle[dir1_name]['mean']:.6e}, max={middle[dir1_name]['max']:.6e}")
            print(f"  {dir2_name}: mean={middle[dir2_name]['mean']:.6e}, max={middle[dir2_name]['max']:.6e}")
            
            # Late layers
            late_start_idx = 2*len(pair_trends['layers'])//3
            print(f"\nLate Layers ({pair_trends['layers'][late_start_idx]}-{pair_trends['layers'][-1]}):")
            late = pair_trends['late']
            print(f"  {dir1_name}: mean={late[dir1_name]['mean']:.6e}, max={late[dir1_name]['max']:.6e}")
            print(f"  {dir2_name}: mean={late[dir2_name]['mean']:.6e}, max={late[dir2_name]['max']:.6e}")
            
            # Trend direction (correlation with layer index)
            print(f"\nTrend Direction (correlation with layer index):")
            corr = pair_trends.get('correlation', {})
            if dir1_name in corr:
                corr_val = corr[dir1_name]
                trend_desc = "increasing" if corr_val > 0.3 else "decreasing" if corr_val < -0.3 else "stable"
                print(f"  {dir1_name}: {trend_desc} (corr={corr_val:.3f})")
            if dir2_name in corr:
                corr_val = corr[dir2_name]
                trend_desc = "increasing" if corr_val > 0.3 else "decreasing" if corr_val < -0.3 else "stable"
                print(f"  {dir2_name}: {trend_desc} (corr={corr_val:.3f})")
            
            # Compare early vs late
            print(f"\nEarly vs Late Comparison:")
            if early[dir1_name]['mean'] > 0 or late[dir1_name]['mean'] > 0:
                early_late_ratio = late[dir1_name]['mean'] / early[dir1_name]['mean'] if early[dir1_name]['mean'] > 1e-12 else float('inf')
                if early_late_ratio != float('inf') and not np.isnan(early_late_ratio):
                    print(f"  {dir1_name}: {early_late_ratio:.2f}x change (late/early)")
                else:
                    print(f"  {dir1_name}: N/A (early is zero)")
            
            if early[dir2_name]['mean'] > 0 or late[dir2_name]['mean'] > 0:
                early_late_ratio = late[dir2_name]['mean'] / early[dir2_name]['mean'] if early[dir2_name]['mean'] > 1e-12 else float('inf')
                if early_late_ratio != float('inf') and not np.isnan(early_late_ratio):
                    print(f"  {dir2_name}: {early_late_ratio:.2f}x change (late/early)")
                else:
                    print(f"  {dir2_name}: N/A (early is zero)")


def main():
    parser = argparse.ArgumentParser(description="Analyze directional differences in saliency flows")
    parser.add_argument("--input", type=str, required=True, help="Path to vis_flows.json file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (optional)")
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.input}")
    data = load_vis_flow_json(args.input)
    print(f"Loaded {len(data)} layers")
    
    results = analyze_directional_differences(data)
    trends = analyze_layer_wise_trends(data)
    
    print_results(results, trends)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            'summary': results,
            'layer_wise_trends': trends
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

