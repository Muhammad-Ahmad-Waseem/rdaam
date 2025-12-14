#!/usr/bin/env python3
"""
Quantitative Analysis Script for Reverse DAAM

Evaluates the performance of the DAAM model in connecting words to bounding boxes
by calculating attention scores for annotated image regions.

Usage:
    cd /path/to/parent/directory
    python -m rdaam.scripts.quantitative_analysis
    
    Or from scripts directory:
    cd rdaam/scripts
    python quantitative_analysis.py
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import rdaam package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rdaam import run_reverse_daam


def calculate_bbox_score(token_scores, decoded_tokens_plot, key_token):
    """
    Calculate the ratio of key_token score to maximum score.
    
    Args:
        token_scores: numpy array of attention scores
        decoded_tokens_plot: List of token strings
        key_token: The key token to evaluate
        
    Returns:
        float: Ratio of key_token score to max score, or 0 if key_token not found
    """
    # Find the index of the key_token in decoded_tokens_plot
    # Handle multi-word tokens
    key_token_indices = []
    
    # Try exact match first
    for idx, token in enumerate(decoded_tokens_plot):
        if token.strip().lower() == key_token.lower():
            key_token_indices.append(idx)
    
    # If no exact match, try partial match for multi-word tokens
    if not key_token_indices:
        key_words = key_token.lower().split()
        for key_word in key_words:
            for idx, token in enumerate(decoded_tokens_plot):
                if key_word in token.strip().lower():
                    key_token_indices.append(idx)
                    break
    
    if not key_token_indices:
        print(f"  Warning: key_token '{key_token}' not found in tokens: {decoded_tokens_plot}")
        return 0.0
    
    # Get the max score for the key_token (if multiple matches)
    key_token_score = max([token_scores[idx] for idx in key_token_indices])
    
    # Get the maximum score overall
    max_score = np.max(token_scores)
    
    if max_score == 0:
        return 0.0
    
    # Calculate ratio
    ratio = key_token_score / max_score
    
    return ratio


def analyze_annotations(json_file, model_id, generation_seed, generation_steps):
    """
    Analyze all annotations in the JSON file.
    
    Args:
        json_file: Path to annotated JSON file
        model_id: Model ID used for generation
        generation_seed: Seed used for generation
        generation_steps: Number of steps used for generation
        
    Returns:
        dict: Analysis results containing scores and statistics
    """
    print(f"Loading annotations from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotated_images = data['annotated_images']
    print(f"Found {len(annotated_images)} annotated images")
    print(f"Total annotations: {data['metadata']['total_annotations']}\n")
    
    # Storage for results
    all_bbox_scores = []  # List of (caption_id, bbox_score) tuples
    caption_scores = {}  # Dict mapping caption_id to list of bbox scores
    
    # Process each annotated image
    for img_idx, img_data in enumerate(tqdm(annotated_images, desc="Analyzing captions")):
        caption_id = img_data['caption_id']
        caption = img_data['caption']
        annotations = img_data['annotations']
        
        print(f"\n[{img_idx+1}/{len(annotated_images)}] Caption ID: {caption_id}")
        print(f"Caption: {caption}")
        print(f"Annotations: {len(annotations)}")
        
        # Storage for this caption's bbox scores
        caption_bbox_scores = []
        
        # Process each bbox annotation
        for ann_idx, annotation in enumerate(annotations):
            bbox = annotation['bounding_box']
            key_token = annotation['key_token']
            
            # Convert bbox dict to list [x_min, y_min, x_max, y_max]
            bbox_list = [
                int(bbox['x_min']),
                int(bbox['y_min']),
                int(bbox['x_max']),
                int(bbox['y_max'])
            ]
            
            print(f"  BBox {ann_idx+1}/{len(annotations)}: {key_token} @ {bbox_list}")
            
            # Run reverse DAAM
            try:
                _, decoded_tokens_plot, token_scores = run_reverse_daam(
                    prompt=caption,
                    seed=generation_seed,
                    steps=generation_steps,
                    model_id=model_id,
                    bbox=bbox_list
                )
                
                # Calculate score for this bbox
                bbox_score = calculate_bbox_score(token_scores, decoded_tokens_plot, key_token)
                
                print(f"    Score: {bbox_score:.4f}")
                
                # Store results
                all_bbox_scores.append((caption_id, bbox_score))
                caption_bbox_scores.append(bbox_score)
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        # Store average score for this caption
        if caption_bbox_scores:
            caption_avg_score = np.mean(caption_bbox_scores)
            caption_scores[caption_id] = caption_avg_score
            print(f"  Caption avg score: {caption_avg_score:.4f}")
    
    # Compute overall statistics
    overall_scores = [score for _, score in all_bbox_scores]
    overall_avg = np.mean(overall_scores) if overall_scores else 0.0
    overall_std = np.std(overall_scores) if overall_scores else 0.0
    
    results = {
        'all_bbox_scores': all_bbox_scores,
        'caption_scores': caption_scores,
        'overall_avg': overall_avg,
        'overall_std': overall_std,
        'total_bboxes': len(all_bbox_scores)
    }
    
    return results


def plot_results(results, output_dir='.'):
    """
    Create visualizations of the analysis results.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save plots
    """
    caption_scores = results['caption_scores']
    all_bbox_scores = results['all_bbox_scores']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Distribution of scores across caption IDs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of caption scores
    caption_ids = sorted(caption_scores.keys())
    scores = [caption_scores[cid] for cid in caption_ids]
    
    ax1.bar(range(len(caption_ids)), scores, color='skyblue', alpha=0.7)
    ax1.axhline(y=results['overall_avg'], color='r', linestyle='--', 
                label=f"Overall Avg: {results['overall_avg']:.4f}")
    ax1.set_xlabel('Caption Index')
    ax1.set_ylabel('Average Score (key_token / max_score)')
    ax1.set_title('Word-BBox Alignment Score per Caption')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of all bbox scores
    bbox_scores = [score for _, score in all_bbox_scores]
    ax2.hist(bbox_scores, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(x=results['overall_avg'], color='r', linestyle='--',
                label=f"Mean: {results['overall_avg']:.4f}")
    ax2.set_xlabel('Score (key_token / max_score)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of BBox Scores (n={results["total_bboxes"]})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'quantitative_analysis_results.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved visualization to: {output_file}")
    
    # Save scores to JSON
    scores_data = {
        'caption_scores': {str(k): float(v) for k, v in caption_scores.items()},
        'overall_statistics': {
            'mean': float(results['overall_avg']),
            'std': float(results['overall_std']),
            'total_bboxes': results['total_bboxes']
        }
    }
    
    json_output = output_path / 'quantitative_scores.json'
    with open(json_output, 'w') as f:
        json.dump(scores_data, f, indent=2)
    print(f"Saved scores to: {json_output}")


def main():
    # Configuration (matching generate_coco_images.py defaults)
    json_file = '../coco_captions_2_3_100_annotated.json'
    model_id = 'CompVis/stable-diffusion-v1-4'
    generation_seed = 42
    generation_steps = 50  # From generate_coco_images.py
    
    print("="*60)
    print("Quantitative Analysis for Reverse DAAM")
    print("="*60)
    print(f"Annotated JSON: {json_file}")
    print(f"Model: {model_id}")
    print(f"Generation Seed: {generation_seed}")
    print(f"Generation Steps: {generation_steps}")
    print("="*60)
    
    # Run analysis
    results = analyze_annotations(json_file, model_id, generation_seed, generation_steps)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total BBoxes Analyzed: {results['total_bboxes']}")
    print(f"Overall Average Score: {results['overall_avg']:.4f} ± {results['overall_std']:.4f}")
    print(f"Number of Captions: {len(results['caption_scores'])}")
    print("="*60)
    
    # Plot results
    plot_results(results)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
