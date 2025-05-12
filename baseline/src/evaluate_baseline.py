import argparse
import json
import pandas as pd
import re
from collections import defaultdict
from typing import Dict, Set, List, Any

def calculate_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 score.
    
    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        
    Returns:
        Dict containing accuracy, precision, recall, and F1 score
    """
    # Avoid division by zero
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3)
    }

def map_r_label_to_req_number(r_label: str) -> str:
    """Map R-label to requirement number.
    
    Args:
        r_label: The R-label (e.g. '7' from 'R7')
        
    Returns:
        The corresponding requirement number (1-19)
    """
    try:
        # Convert R7-R25 to req 1-19 (subtract 6)
        r_number = int(r_label)
        req_number = r_number - 6
        
        return str(req_number) if req_number > 0 else r_label
    except ValueError:
        # If not a number, return as is
        return r_label

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline DPA completeness metrics")
    parser.add_argument("--baseline_results", type=str, required=True,
                      help="Path to baseline results JSON file")
    parser.add_argument("--dpa", type=str, required=True,
                      help="Path to DPA segments CSV file")
    parser.add_argument("--output", type=str, required=True,
                      help="Output file for evaluation metrics")
    parser.add_argument("--detailed", action="store_true",
                      help="Print detailed metrics for each requirement")
    
    args = parser.parse_args()
    
    # Load baseline results
    print(f"Loading baseline results from: {args.baseline_results}")
    with open(args.baseline_results, 'r') as f:
        baseline_results = json.load(f)
    
    # Load DPA data for ground truth
    print(f"Loading DPA data from: {args.dpa}")
    df = pd.read_csv(args.dpa)
    
    target_dpa = baseline_results['dpa_name']
    print(f"Evaluating DPA: {target_dpa}")
    
    # Filter for the target DPA
    dpa_segments = df[df['DPA'] == target_dpa]
    
    if dpa_segments.empty:
        print(f"Error: No segments found for DPA '{target_dpa}'")
        return
    
    # Get segment IDs from baseline results (they're 0-based indices)
    segment_indices = list(range(baseline_results['metrics']['total_segments']))
    
    # Map segment indices to actual segment IDs from the CSV
    segment_ids = dpa_segments['ID'].astype(str).tolist()[:len(segment_indices)]
    
    # Create a mapping of segments to their ground truth requirements
    segment_to_ground_truth_reqs = defaultdict(set)
    for _, row in dpa_segments.iterrows():
        if str(row['ID']) not in segment_ids:
            continue
            
        target = row['target']
        if target and target != 'other':
            # Extract requirement numbers from target column (format: R7, R8, etc.)
            req_matches = re.findall(r'R(\d+)', target)
            for req in req_matches:
                # Convert R7-R25 to 1-19 by subtracting 6
                req_number = str(int(req) - 6)
                segment_to_ground_truth_reqs[str(row['ID'])].add(req_number)
    
    # Create mapping of requirements to segments that satisfy them
    req_to_ground_truth_segments = defaultdict(set)
    for segment_id, reqs in segment_to_ground_truth_reqs.items():
        for req_id in reqs:
            req_to_ground_truth_segments[req_id].add(segment_id)
    
    # Get the set of requirements evaluated in baseline
    baseline_req_ids = list(baseline_results['requirements'].keys())
    print(f"Evaluating {len(baseline_req_ids)} requirements: {', '.join(baseline_req_ids[:5])}{'...' if len(baseline_req_ids) > 5 else ''}")
    
    # Create mapping of ground truth for each requirement
    all_ground_truth_reqs = set()
    for req_id in baseline_req_ids:
        if req_id in req_to_ground_truth_segments and req_to_ground_truth_segments[req_id]:
            all_ground_truth_reqs.add(req_id)
    
    # Get the set of satisfied requirements from baseline results
    satisfied_reqs = set()
    for req_id, req_data in baseline_results['requirements'].items():
        if req_data['satisfied']:
            satisfied_reqs.add(req_id)
    
    # Determine completeness based on the requirements
    ground_truth_complete = len(set(baseline_req_ids) - all_ground_truth_reqs) == 0
    predicted_complete = len(set(baseline_req_ids) - satisfied_reqs) == 0
    
    # Calculate metrics at the requirement level
    req_tp, req_fp, req_tn, req_fn = 0, 0, 0, 0
    
    for req_id in baseline_req_ids:
        ground_truth_satisfied = req_id in all_ground_truth_reqs
        predicted_satisfied = req_id in satisfied_reqs
        
        if ground_truth_satisfied and predicted_satisfied:
            req_tp += 1
        elif ground_truth_satisfied and not predicted_satisfied:
            req_fn += 1
        elif not ground_truth_satisfied and predicted_satisfied:
            req_fp += 1
        else:  # not ground_truth_satisfied and not predicted_satisfied
            req_tn += 1
    
    requirement_metrics = calculate_metrics(req_tp, req_fp, req_tn, req_fn)
    
    # Calculate metrics at the segment level for each requirement
    segment_metrics = defaultdict(dict)
    total_seg_tp, total_seg_fp, total_seg_tn, total_seg_fn = 0, 0, 0, 0
    
    # First, map baseline segment indices to actual segment IDs
    index_to_segment_id = {idx: seg_id for idx, seg_id in enumerate(segment_ids)}
    
    for req_id in baseline_req_ids:
        seg_tp, seg_fp, seg_tn, seg_fn = 0, 0, 0, 0
        
        # Get baseline predictions for this requirement
        satisfying_indices = baseline_results['requirements'][req_id]['satisfying_segments']
        predicted_segment_ids = [index_to_segment_id[idx] for idx in satisfying_indices]
        
        # Count ground truth segments for this requirement
        ground_truth_segments = req_to_ground_truth_segments.get(req_id, set())
        
        # For each segment
        for idx, segment_id in index_to_segment_id.items():
            # Ground truth: is this segment satisfying the requirement?
            ground_truth_satisfies = segment_id in ground_truth_segments
            
            # Prediction: did the baseline predict this segment satisfies the requirement?
            predicted_satisfies = idx in satisfying_indices
            
            if ground_truth_satisfies and predicted_satisfies:
                seg_tp += 1
                total_seg_tp += 1
            elif ground_truth_satisfies and not predicted_satisfies:
                seg_fn += 1
                total_seg_fn += 1
            elif not ground_truth_satisfies and predicted_satisfies:
                seg_fp += 1
                total_seg_fp += 1
            else:  # not ground_truth_satisfies and not predicted_satisfies
                seg_tn += 1
                total_seg_tn += 1
        
        # Calculate metrics for this requirement
        metrics = calculate_metrics(seg_tp, seg_fp, seg_tn, seg_fn)
        segment_metrics[req_id] = {
            "metrics": metrics,
            "confusion_matrix": {
                "true_positives": seg_tp,
                "false_positives": seg_fp,
                "true_negatives": seg_tn,
                "false_negatives": seg_fn
            }
        }
    
    # Calculate overall segment-level metrics
    overall_segment_metrics = calculate_metrics(
        total_seg_tp, total_seg_fp, total_seg_tn, total_seg_fn
    )
    
    # Prepare final evaluation results
    evaluation_results = {
        "dpa_name": target_dpa,
        "completeness": {
            "ground_truth_complete": ground_truth_complete,
            "predicted_complete": predicted_complete,
            "correctly_predicted": ground_truth_complete == predicted_complete
        },
        "requirement_level_metrics": {
            "metrics": requirement_metrics,
            "confusion_matrix": {
                "true_positives": req_tp,
                "false_positives": req_fp,
                "true_negatives": req_tn,
                "false_negatives": req_fn
            }
        },
        "segment_level_metrics": {
            "overall": overall_segment_metrics,
            "confusion_matrix": {
                "true_positives": total_seg_tp,
                "false_positives": total_seg_fp,
                "true_negatives": total_seg_tn,
                "false_negatives": total_seg_fn
            },
            "per_requirement": segment_metrics
        }
    }
    
    # Print summary of results
    print("\nEvaluation Results:")
    print(f"DPA: {target_dpa}")
    print(f"Number of requirements: {len(baseline_req_ids)}")
    print(f"Number of segments: {len(segment_ids)}")
    print(f"Ground Truth Complete: {ground_truth_complete}")
    print(f"Predicted Complete: {predicted_complete}")
    print(f"Correctly Predicted Completeness: {ground_truth_complete == predicted_complete}")
    
    print("\nRequirement-Level Metrics:")
    print(f"  Confusion Matrix: TP={req_tp}, FP={req_fp}, TN={req_tn}, FN={req_fn}")
    for metric, value in requirement_metrics.items():
        print(f"  {metric.capitalize()}: {value}")
    
    print("\nSegment-Level Metrics (Overall):")
    print(f"  Confusion Matrix: TP={total_seg_tp}, FP={total_seg_fp}, TN={total_seg_tn}, FN={total_seg_fn}")
    for metric, value in overall_segment_metrics.items():
        print(f"  {metric.capitalize()}: {value}")
    
    # Optionally print detailed metrics per requirement
    if args.detailed:
        print("\nDetailed Metrics Per Requirement:")
        for req_id in baseline_req_ids:
            metrics = segment_metrics[req_id]["metrics"]
            cm = segment_metrics[req_id]["confusion_matrix"]
            print(f"\nRequirement {req_id}:")
            print(f"  Confusion Matrix: TP={cm['true_positives']}, FP={cm['false_positives']}, TN={cm['true_negatives']}, FN={cm['false_negatives']}")
            print(f"  Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1_score']}")
            
            # Print the actual and predicted satisfied segment counts
            ground_truth_segs = len(req_to_ground_truth_segments.get(req_id, set()))
            pred_segs = len(baseline_results['requirements'][req_id]['satisfying_segments'])
            print(f"  Ground Truth Satisfying Segments: {ground_truth_segs}")
            print(f"  Predicted Satisfying Segments: {pred_segs}")
    
    # Save results to output file
    with open(args.output, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nDetailed evaluation results saved to: {args.output}")

if __name__ == "__main__":
    main() 