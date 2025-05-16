#!/usr/bin/env python3
"""
aggregate_paragraph_metrics.py - Aggregate paragraph-level metrics from multiple DPAs
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

def calculate_aggregated_metrics(metrics_list: List[Dict]) -> Dict:
    """Calculate aggregated metrics from a list of metrics dictionaries.
    
    Args:
        metrics_list: List of metrics dictionaries, each having accuracy, precision, recall, and f1_score
        
    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0
        }
    
    # Sum all metrics
    sum_metrics = defaultdict(float)
    for metrics in metrics_list:
        for key, value in metrics.items():
            sum_metrics[key] += value
    
    # Calculate average
    avg_metrics = {k: v / len(metrics_list) for k, v in sum_metrics.items()}
    
    # Round to 3 decimal places
    return {k: round(v, 3) for k, v in avg_metrics.items()}

def aggregate_confusion_matrices(matrices: List[Dict]) -> Dict:
    """Aggregate confusion matrices by summing their values.
    
    Args:
        matrices: List of confusion matrices
        
    Returns:
        Aggregated confusion matrix
    """
    if not matrices:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0
        }
    
    result = defaultdict(int)
    for matrix in matrices:
        for key, value in matrix.items():
            result[key] += value
    
    return dict(result)

def main():
    parser = argparse.ArgumentParser(description="Aggregate paragraph metrics from multiple DPAs")
    parser.add_argument("--input_files", type=str, required=True, help="Space-separated list of input JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output file for aggregated results")
    
    args = parser.parse_args()
    
    # Parse input files
    input_files = []
    if args.input_files:
        # Handle both space-separated and comma-separated lists
        if ',' in args.input_files:
            input_files = [f.strip() for f in args.input_files.split(',')]
        else:
            input_files = [f.strip() for f in args.input_files.split()]
    
    if not input_files:
        print("Error: No input files specified")
        return
    
    print(f"Aggregating paragraph metrics from {len(input_files)} files")
    
    # Load all paragraph metrics
    all_metrics = []
    for file_path in input_files:
        try:
            with open(file_path, 'r') as f:
                all_metrics.append(json.load(f))
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_metrics:
        print("No valid metrics files found. Exiting.")
        return
    
    # Extract DPA names for reference
    dpa_names = [metrics.get("dpa", f"Unknown DPA {i}") for i, metrics in enumerate(all_metrics)]
    
    # Initialize aggregated result structure
    aggregated_result = {
        "dpas": dpa_names,
        "total_evaluations": len(all_metrics),
        "aggregated_metrics": {},
        "per_dpa_results": {}
    }
    
    # Store individual DPA results for reference
    for i, dpa in enumerate(dpa_names):
        aggregated_result["per_dpa_results"][dpa] = all_metrics[i]
    
    # Aggregate binary classification metrics
    binary_metrics = []
    binary_confusion_matrices = []
    total_segments = 0
    
    for metric_result in all_metrics:
        binary_data = metric_result.get("binary_classification", {})
        segment_metrics = binary_data.get("segment_metrics", {})
        binary_metrics.append(segment_metrics.get("metrics", {}))
        binary_confusion_matrices.append(segment_metrics.get("confusion_matrix", {}))
        total_segments += segment_metrics.get("total_segments", 0)
    
    # Store the aggregated binary metrics
    aggregated_result["aggregated_metrics"]["binary_classification"] = {
        "segment_metrics": {
            "metrics": calculate_aggregated_metrics(binary_metrics),
            "confusion_matrix": aggregate_confusion_matrices(binary_confusion_matrices),
            "total_segments": total_segments
        }
    }
    
    # Aggregate multi-class classification metrics
    multiclass_metrics = []
    total_paragraphs = 0
    correct_predictions = 0
    partial_matches = 0
    
    for metric_result in all_metrics:
        multiclass_data = metric_result.get("multiclass_classification", {})
        multiclass_metrics.append({
            "accuracy": multiclass_data.get("accuracy", 0),
            "partial_accuracy": multiclass_data.get("partial_accuracy", 0)
        })
        total_paragraphs += multiclass_data.get("total_paragraphs", 0)
        correct_predictions += multiclass_data.get("correct_predictions", 0)
        partial_matches += multiclass_data.get("partial_matches", 0)
    
    # Calculate average accuracy and partial accuracy
    avg_multiclass_metrics = calculate_aggregated_metrics(multiclass_metrics)
    
    # Store the aggregated multi-class metrics
    aggregated_result["aggregated_metrics"]["multiclass_classification"] = {
        "accuracy": avg_multiclass_metrics.get("accuracy", 0),
        "partial_accuracy": avg_multiclass_metrics.get("partial_accuracy", 0),
        "total_paragraphs": total_paragraphs,
        "correct_predictions": correct_predictions,
        "partial_matches": partial_matches
    }
    
    # Save aggregated results
    with open(args.output, 'w') as f:
        json.dump(aggregated_result, f, indent=2)
    
    # Print summary
    print("\nAggregated Paragraph Metrics Summary:")
    print(f"Total DPAs evaluated: {len(dpa_names)}")
    print(f"DPAs evaluated: {', '.join(dpa_names)}")
    print(f"Total segments evaluated: {total_segments}")
    print(f"Total paragraphs evaluated: {total_paragraphs}")
    
    binary_cm = aggregated_result["aggregated_metrics"]["binary_classification"]["segment_metrics"]["confusion_matrix"]
    binary_metrics = aggregated_result["aggregated_metrics"]["binary_classification"]["segment_metrics"]["metrics"]
    
    print("\nAggregated Binary Classification Metrics (Segment Level):")
    print(f"Confusion Matrix: TP={binary_cm['true_positives']}, FP={binary_cm['false_positives']}, " 
          f"TN={binary_cm['true_negatives']}, FN={binary_cm['false_negatives']}")
    for metric, value in binary_metrics.items():
        print(f"  {metric.capitalize()}: {value}")
    
    multiclass = aggregated_result["aggregated_metrics"]["multiclass_classification"]
    print("\nAggregated Multi-class Classification Metrics (Paragraph Level):")
    print(f"  Accuracy (exact match): {multiclass['accuracy']}")
    print(f"  Accuracy (partial match): {multiclass['partial_accuracy']}")
    print(f"  Correct Predictions: {multiclass['correct_predictions']} of {multiclass['total_paragraphs']} " 
          f"({round(multiclass['correct_predictions'] / multiclass['total_paragraphs'] * 100, 1) if multiclass['total_paragraphs'] > 0 else 0}%)")
    print(f"  Partial Matches: {multiclass['partial_matches']} of {multiclass['total_paragraphs']} " 
          f"({round(multiclass['partial_matches'] / multiclass['total_paragraphs'] * 100, 1) if multiclass['total_paragraphs'] > 0 else 0}%)")
    
    print(f"\nAggregated paragraph metrics saved to: {args.output}")

if __name__ == "__main__":
    main() 