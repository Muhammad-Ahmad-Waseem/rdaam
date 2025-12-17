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
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Import rdaam batch analyzer
from rdaam.analysis.batch import ReverseDAAMBatch

# Setup logger
logger = logging.getLogger(__name__)


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
        logger.warning(f"key_token '{key_token}' not found in tokens: {decoded_tokens_plot}")
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
    logger.info(f"Loading annotations from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotated_images = data['annotated_images']
    logger.info(f"Found {len(annotated_images)} annotated images")
    logger.info(f"Total annotations: {data['metadata']['total_annotations']}")
    
    # Initialize batch analyzer ONCE
    logger.info(f"Initializing batch analyzer: {model_id}")
    analyzer = ReverseDAAMBatch(model_id=model_id)
    logger.info("Batch analyzer ready")
    
    # Storage for results
    all_bbox_scores = []  # List of (caption_id, bbox_score) tuples
    caption_scores = {}  # Dict mapping caption_id to list of bbox scores
    
    # Process each annotated image
    for img_idx, img_data in enumerate(tqdm(annotated_images, desc="Analyzing captions")):
        caption_id = img_data['caption_id']
        caption = img_data['caption']
        annotations = img_data['annotations']
        
        logger.info(f"[{img_idx+1}/{len(annotated_images)}] Caption ID: {caption_id}")
        logger.debug(f"Caption: {caption}")
        logger.debug(f"Annotations: {len(annotations)}")
        
        # Generate global map ONCE per caption
        try:
            global_map, decoded_tokens, _ = analyzer.generate_map(
                prompt=caption,
                seed=generation_seed,
                steps=generation_steps
            )
        except Exception as e:
            logger.error(f"Error generating map for caption {caption_id}: {e}")
            continue
        
        # Storage for this caption's bbox scores
        caption_bbox_scores = []
        
        # Process each bbox annotation using the SAME global map
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
            
            logger.debug(f"BBox {ann_idx+1}/{len(annotations)}: {key_token} @ {bbox_list}")
            
            # Extract bbox scores from pre-generated map (FAST!)
            try:
                token_scores, decoded_tokens_plot = analyzer.extract_bbox_scores(
                    global_map, decoded_tokens, bbox_list
                )
                
                # Calculate score for this bbox
                bbox_score = calculate_bbox_score(token_scores, decoded_tokens_plot, key_token)
                
                logger.debug(f"Score: {bbox_score:.4f}")
                
                # Store results
                all_bbox_scores.append((caption_id, bbox_score))
                caption_bbox_scores.append(bbox_score)
                
            except Exception as e:
                logger.error(f"Error processing bbox: {e}")
                continue
        
        # Store average score for this caption
        if caption_bbox_scores:
            caption_avg_score = np.mean(caption_bbox_scores)
            caption_scores[caption_id] = caption_avg_score
            logger.info(f"Caption avg score: {caption_avg_score:.4f}")
    
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
    logger.info(f"Saved visualization to: {output_file}")
    
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
    logger.info(f"Saved scores to: {json_output}")


def main():
    # Setup logging - suppress all output including rdaam package logs
    logging.basicConfig(
        level=logging.CRITICAL,  # Only show critical errors
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress rdaam package logging
    logging.getLogger('rdaam').setLevel(logging.CRITICAL)
    logging.getLogger('rdaam.analysis').setLevel(logging.CRITICAL)
    logging.getLogger('rdaam.analysis.reverse').setLevel(logging.CRITICAL)
    logging.getLogger('rdaam.core').setLevel(logging.CRITICAL)
    
    # Configuration (matching generate_coco_images.py defaults)
    json_file = 'coco_captions_2_3_100_annotated.json'
    model_id = 'CompVis/stable-diffusion-v1-4'
    generation_seed = 42
    generation_steps = 50  # From generate_coco_images.py
    
    # Run analysis silently (only tqdm progress bar will show)
    results = analyze_annotations(json_file, model_id, generation_seed, generation_steps)
    
    # Plot results
    plot_results(results)


if __name__ == '__main__':
    main()
