#!/usr/bin/env python3
"""
aggregate_evaluations.py - Aggregate evaluation results from multiple DPAs
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
    parser = argparse.ArgumentParser(description="Aggregate evaluation results from multiple DPAs")
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
    
    print(f"Aggregating evaluation results from {len(input_files)} files")
    
    # Load all evaluation results
    all_evaluations = []
    for file_path in input_files:
        try:
            with open(file_path, 'r') as f:
                all_evaluations.append(json.load(f))
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_evaluations:
        print("No valid evaluation files found. Exiting.")
        return
    
    # Extract DPA names for reference
    dpa_names = [eval_result.get("dpa", f"Unknown DPA {i}") for i, eval_result in enumerate(all_evaluations)]
    
    # Initialize aggregated result structure
    aggregated_result = {
        "dpas": dpa_names,
        "total_evaluations": len(all_evaluations),
        "segments_evaluated": sum(eval_result.get("segments_evaluated", 0) for eval_result in all_evaluations),
        "requirements_evaluated": all_evaluations[0].get("requirements_evaluated", []),
        "aggregated_metrics": {},
        "per_dpa_results": {}
    }
    
    # Store individual DPA results for reference
    for i, dpa in enumerate(dpa_names):
        aggregated_result["per_dpa_results"][dpa] = all_evaluations[i]
    
    # Aggregate requirements metrics
    req_metrics_list = [eval_result.get("requirements_metrics", {}).get("overall", {}) 
                       for eval_result in all_evaluations]
    req_confusion_matrices = [eval_result.get("requirements_metrics", {}).get("confusion_matrix", {}) 
                             for eval_result in all_evaluations]
    
    aggregated_result["aggregated_metrics"]["requirements"] = {
        "overall": calculate_aggregated_metrics(req_metrics_list),
        "confusion_matrix": aggregate_confusion_matrices(req_confusion_matrices)
    }
    
    # Aggregate segment metrics
    seg_metrics_list = [eval_result.get("segment_metrics", {}).get("overall", {}) 
                       for eval_result in all_evaluations]
    seg_confusion_matrices = [eval_result.get("segment_metrics", {}).get("confusion_matrix", {}) 
                             for eval_result in all_evaluations]
    
    aggregated_result["aggregated_metrics"]["segments"] = {
        "overall": calculate_aggregated_metrics(seg_metrics_list),
        "confusion_matrix": aggregate_confusion_matrices(seg_confusion_matrices)
    }
    
    # Compute aggregated completeness
    complete_ground_truth = sum(1 for eval_result in all_evaluations 
                               if eval_result.get("ground_truth", {}).get("is_complete", False))
    complete_prediction = sum(1 for eval_result in all_evaluations 
                             if eval_result.get("prediction", {}).get("is_complete", False))
    
    aggregated_result["completeness_summary"] = {
        "ground_truth_complete": complete_ground_truth,
        "prediction_complete": complete_prediction,
        "ground_truth_percentage": round(complete_ground_truth / len(all_evaluations) * 100, 1),
        "prediction_percentage": round(complete_prediction / len(all_evaluations) * 100, 1),
    }
    
    # Save aggregated results
    with open(args.output, 'w') as f:
        json.dump(aggregated_result, f, indent=2)
    
    # Print summary
    print("\nAggregated Evaluation Summary:")
    print(f"Total DPAs evaluated: {len(dpa_names)}")
    print(f"DPAs evaluated: {', '.join(dpa_names)}")
    print(f"Total segments evaluated: {aggregated_result['segments_evaluated']}")
    
    req_cm = aggregated_result["aggregated_metrics"]["requirements"]["confusion_matrix"]
    print("\nAggregated Requirement-Level Metrics:")
    print(f"Confusion Matrix: TP={req_cm['true_positives']}, FP={req_cm['false_positives']}, " 
          f"TN={req_cm['true_negatives']}, FN={req_cm['false_negatives']}")
    
    for metric, value in aggregated_result["aggregated_metrics"]["requirements"]["overall"].items():
        print(f"  {metric.capitalize()}: {value}")
    
    seg_cm = aggregated_result["aggregated_metrics"]["segments"]["confusion_matrix"]
    print("\nAggregated Segment-Level Metrics:")
    print(f"Confusion Matrix: TP={seg_cm['true_positives']}, FP={seg_cm['false_positives']}, " 
          f"TN={seg_cm['true_negatives']}, FN={seg_cm['false_negatives']}")
    
    for metric, value in aggregated_result["aggregated_metrics"]["segments"]["overall"].items():
        print(f"  {metric.capitalize()}: {value}")
    
    print(f"\nCompleteness Summary:")
    comp_summary = aggregated_result["completeness_summary"]
    print(f"  Ground Truth Complete: {comp_summary['ground_truth_complete']} of {len(all_evaluations)} " 
          f"({comp_summary['ground_truth_percentage']}%)")
    print(f"  Prediction Complete: {comp_summary['prediction_complete']} of {len(all_evaluations)} " 
          f"({comp_summary['prediction_percentage']}%)")
    
    print(f"\nAggregated results saved to: {args.output}")

if __name__ == "__main__":
    main() 