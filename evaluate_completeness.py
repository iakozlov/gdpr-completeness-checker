import os
import json
import argparse
import pandas as pd
import re
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Any

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
        r_label: The R-label (e.g. '10' from 'R10')
        
    Returns:
        The corresponding requirement number
    """
    # Convert R10-R29 to req 1-20 (subtract 9)
    # Note: R14 is not present in the dataset
    r_number = int(r_label)
    
    # Adjust for the absence of R14
    if r_number < 14:
        # R10-R13 map to 1-4
        req_number = r_number - 9
    elif r_number == 14:
        # R14 doesn't exist
        req_number = -1  # Invalid
    else:
        # R15-R29 map to 5-19
        req_number = r_number - 10
    
    return str(req_number) if req_number > 0 else r_label

def main():
    parser = argparse.ArgumentParser(description="Evaluate DPA completeness based on Deolingo results")
    parser.add_argument("--results", type=str, default="results/deolingo_results.txt",
                        help="Path to Deolingo results file")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--output", type=str, default="results/evaluation_results.json",
                        help="Output file for evaluation results")
    parser.add_argument("--target_dpa", type=str, default="Online 1",
                        help="Target DPA to evaluate (default: Online 1)")
    parser.add_argument("--req_ids", type=str, default="all",
                        help="Comma-separated list of requirement IDs to process, or 'all' (default: all)")
    parser.add_argument("--max_segments", type=int, default=0,
                        help="Maximum number of segments to process (0 means all, default: 0)")
    args = parser.parse_args()

    target_dpa = args.target_dpa
    
    # Step 1: Load required data
    print(f"Loading DPA segments from: {args.dpa}")
    df = pd.read_csv(args.dpa)
    
    # Filter for the target DPA
    dpa_segments = df[df['DPA'] == target_dpa]
    
    if dpa_segments.empty:
        print(f"Error: No segments found for DPA '{target_dpa}'")
        return

    # Apply segment limit if specified
    if args.max_segments > 0:
        dpa_segments = dpa_segments.head(args.max_segments)
        print(f"Using first {len(dpa_segments)} segments for DPA: {target_dpa}")
    else:
        print(f"Using all {len(dpa_segments)} segments for DPA: {target_dpa}")
    
    # Set of segment IDs to consider in the evaluation
    segment_ids = set(dpa_segments['ID'].astype(str))
    
    # Step 2: Parse requirement IDs
    if args.req_ids.lower() != "all":
        req_ids = set(id.strip() for id in args.req_ids.split(","))
        print(f"Evaluating requirements: {', '.join(req_ids)}")
    else:
        # Default to requirements R10-R29 if "all" is specified (19 requirements total, excluding R14)
        req_ids = set(str(i) for i in range(10, 14)) | set(str(i) for i in range(15, 30))
        print(f"Evaluating all requirements (R10-R13, R15-R29)")
    
    # Step 3: Load and parse Deolingo results
    print(f"\nProcessing Deolingo results from: {args.results}")
    with open(args.results, 'r') as f:
        results_text = f.read()
    
    # Step 4: Parse results - split by dashed lines
    results = defaultdict(dict)
    sections = results_text.split('--------------------------------------------------')
    
    for section in sections:
        if not section.strip():
            continue
            
        # Extract requirement ID and segment ID from the first line
        first_line = section.strip().split('\n')[0].strip()
        req_match = re.search(r'Requirement\s+(\d+)', first_line)
        seg_match = re.search(r'Segment\s+(\d+)', first_line)
        
        if not req_match or not seg_match:
            continue
            
        req_id = req_match.group(1)
        segment_id = seg_match.group(1)
        
        # Only process segments and requirements that match our filters
        if (req_id not in req_ids) or (segment_id not in segment_ids):
            continue
        
        # Look for the status in the FACTS line
        status = 'not_mentioned'  # Default status
        
        if 'Error processing file' in section:
            status = 'not_mentioned'
        else:
            # Check for FACTS line with status
            for line in section.strip().split('\n'):
                if line.startswith('FACTS:'):
                    status_match = re.search(r'status\((\w+)\)', line)
                    if status_match:
                        status = status_match.group(1)
        
        # Store the status
        results[req_id][segment_id] = status
    
    # Create a mapping of segments to their ground truth requirements
    # Note: This maps segment IDs to sets of R-labels (e.g. '10', '11', etc.)
    segment_to_ground_truth_reqs = defaultdict(set)
    for _, row in dpa_segments.iterrows():
        segment_id = str(row['ID'])
        target = row['target']
        if target and target != 'other':
            req_matches = re.findall(r'R(\d+)', target)
            for req in req_matches:
                if req in req_ids:
                    segment_to_ground_truth_reqs[segment_id].add(req)
    
    # Create a mapping of requirements to segments that satisfy them
    # This will be used for segment-level metrics and requirement satisfaction
    req_to_ground_truth_segments = defaultdict(set)
    for segment_id, reqs in segment_to_ground_truth_reqs.items():
        for req_id in reqs:
            req_to_ground_truth_segments[req_id].add(segment_id)
    
    # Step 5: Get ground truth from target column for filtered segments
    all_ground_truth_reqs = set()
    for req_id, segments in req_to_ground_truth_segments.items():
        if segments:  # If any segments satisfy this requirement
            all_ground_truth_reqs.add(req_id)
    
    # Check which requirements are satisfied in the results
    satisfied_reqs = set()
    for req_id, segments in results.items():
        is_satisfied = any(status == "satisfied" for status in segments.values())
        if is_satisfied:
            satisfied_reqs.add(req_id)
    
    # Determine completeness based on the filtered requirements
    filtered_required_reqs = req_ids
    
    ground_truth_complete = len(filtered_required_reqs - all_ground_truth_reqs) == 0
    predicted_complete = len(filtered_required_reqs - satisfied_reqs) == 0
    
    # Calculate metrics at the requirement level
    req_tp, req_fp, req_tn, req_fn = 0, 0, 0, 0
    
    for req_id in filtered_required_reqs:
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
    
    # Validate the confusion matrix totals
    total_req_count = req_tp + req_fp + req_tn + req_fn
    if total_req_count != len(filtered_required_reqs):
        print(f"\nWarning: Confusion matrix counts ({total_req_count}) don't match requirement count ({len(filtered_required_reqs)})")
        print(f"TP={req_tp}, FP={req_fp}, TN={req_tn}, FN={req_fn}")
    
    requirement_metrics = calculate_metrics(req_tp, req_fp, req_tn, req_fn)
    
    # Calculate metrics at the segment level for each requirement
    # We'll track the mapping between R-labels and requirement numbers
    r_to_req_mapping = {r: map_r_label_to_req_number(r) for r in filtered_required_reqs}
    
    segment_metrics = defaultdict(dict)
    total_seg_tp, total_seg_fp, total_seg_tn, total_seg_fn = 0, 0, 0, 0
    
    for req_id in filtered_required_reqs:
        seg_tp, seg_fp, seg_tn, seg_fn = 0, 0, 0, 0
        
        # Get the mapped requirement number for reporting
        req_number = r_to_req_mapping[req_id]
        
        # For each segment
        for segment_id in segment_ids:
            # Ground truth: is this segment satisfying the requirement?
            ground_truth_satisfies = segment_id in req_to_ground_truth_segments.get(req_id, set())
            
            # Prediction: did we predict this segment satisfies the requirement?
            predicted_satisfies = (segment_id in results.get(req_id, {}) and 
                                  results[req_id].get(segment_id) == "satisfied")
            
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
    
    # Calculate overall segment metrics
    overall_segment_metrics = calculate_metrics(
        total_seg_tp, total_seg_fp, total_seg_tn, total_seg_fn
    )
    
    # Compute evaluation metrics
    evaluation = {
        "dpa": target_dpa,
        "segments_evaluated": len(segment_ids),
        "requirements_evaluated": sorted(list(req_ids)),
        "ground_truth": {
            "is_complete": ground_truth_complete,
            "requirements": {r: r in all_ground_truth_reqs for r in filtered_required_reqs},
            "missing_requirements": sorted(list(filtered_required_reqs - all_ground_truth_reqs))
        },
        "prediction": {
            "is_complete": predicted_complete,
            "requirements": {r: r in satisfied_reqs for r in filtered_required_reqs},
            "missing_requirements": sorted(list(filtered_required_reqs - satisfied_reqs))
        },
        "agreement": ground_truth_complete == predicted_complete,
        "requirements_metrics": {
            "overall": requirement_metrics,
            "confusion_matrix": {
                "true_positives": req_tp,
                "false_positives": req_fp,
                "true_negatives": req_tn,
                "false_negatives": req_fn
            }
        },
        "segment_metrics": {
            "overall": overall_segment_metrics,
            "by_requirement": segment_metrics,
            "confusion_matrix": {
                "true_positives": total_seg_tp,
                "false_positives": total_seg_fp,
                "true_negatives": total_seg_tn,
                "false_negatives": total_seg_fn
            }
        },
        "requirements_details": {}
    }
    
    # Add detailed requirement information
    for req_id in sorted(filtered_required_reqs):
        ground_truth_satisfied = req_id in all_ground_truth_reqs
        predicted_satisfied = req_id in satisfied_reqs
        
        satisfied_segments = []
        if req_id in results:
            satisfied_segments = [seg for seg, status in results[req_id].items() if status == "satisfied"]
        
        req_number = r_to_req_mapping.get(req_id, req_id)
        
        evaluation["requirements_details"][req_id] = {
            "requirement_number": req_number,  # Mapped requirement number (R10->1, R15->5, etc.)
            "ground_truth": ground_truth_satisfied,
            "prediction": predicted_satisfied,
            "agreement": ground_truth_satisfied == predicted_satisfied,
            "satisfied_segments": sorted(satisfied_segments),
            "ground_truth_segments": sorted(list(req_to_ground_truth_segments.get(req_id, set())))
        }
    
    # Save results
    print(f"\nSaving evaluation results to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"DPA: {target_dpa}")
    print(f"Number of segments evaluated: {len(segment_ids)}")
    print(f"Number of requirements evaluated: {len(filtered_required_reqs)}")
    print(f"Ground Truth Label: {'Complete' if ground_truth_complete else 'Not Complete'}")
    print(f"Predicted Label: {'Complete' if predicted_complete else 'Not Complete'}")
    print(f"Labels Match: {'Yes' if evaluation['agreement'] else 'No'}")
    
    # Print requirement details
    print("\nRequirement Details:")
    for req_id in sorted(filtered_required_reqs):
        detail = evaluation["requirements_details"][req_id]
        req_number = r_to_req_mapping.get(req_id, req_id)
        print(f"\nRequirement R{req_id} (req_{req_number}):")
        print(f"  Ground Truth: {'Satisfied' if detail['ground_truth'] else 'Not Satisfied'}")
        print(f"  Prediction: {'Satisfied' if detail['prediction'] else 'Not Satisfied'}")
        print(f"  Match: {'Yes' if detail['agreement'] else 'No'}")
        if detail['satisfied_segments']:
            print(f"  Predicted satisfied in segments: {', '.join(detail['satisfied_segments'])}")
        else:
            print("  Not predicted satisfied in any segment")
        if detail['ground_truth_segments']:
            print(f"  Ground truth satisfied in segments: {', '.join(detail['ground_truth_segments'])}")
        
        # Print segment metrics for this requirement
        req_segment_metrics = segment_metrics[req_id]['metrics']
        print(f"  Segment Metrics for R{req_id}:")
        print(f"    Accuracy:  {req_segment_metrics['accuracy']}")
        print(f"    Precision: {req_segment_metrics['precision']}")
        print(f"    Recall:    {req_segment_metrics['recall']}")
        print(f"    F1 Score:  {req_segment_metrics['f1_score']}")
    
    # Print metrics at the end
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY")
    print("="*50)
    
    print("\nRequirement-Level Metrics:")
    print(f"  Total Requirements Evaluated: {len(filtered_required_reqs)}")
    print(f"  Accuracy:  {requirement_metrics['accuracy']}")
    print(f"  Precision: {requirement_metrics['precision']}")
    print(f"  Recall:    {requirement_metrics['recall']}")
    print(f"  F1 Score:  {requirement_metrics['f1_score']}")
    print(f"  Confusion Matrix: TP={req_tp}, FP={req_fp}, TN={req_tn}, FN={req_fn}")
    
    print("\nSegment-Level Metrics:")
    print(f"  Total Segment-Requirement Pairs: {total_seg_tp + total_seg_fp + total_seg_tn + total_seg_fn}")
    print(f"  Accuracy:  {overall_segment_metrics['accuracy']}")
    print(f"  Precision: {overall_segment_metrics['precision']}")
    print(f"  Recall:    {overall_segment_metrics['recall']}")
    print(f"  F1 Score:  {overall_segment_metrics['f1_score']}")
    print(f"  Confusion Matrix: TP={total_seg_tp}, FP={total_seg_fp}, TN={total_seg_tn}, FN={total_seg_fn}")

if __name__ == "__main__":
    main()