#!/usr/bin/env python3
"""
paragraph_metrics.py - Calculate metrics for paragraph-level analysis of DPA completeness
Implementing binary classification (provision x vs. not provision x) and 
multi-class classification (each provision is one class)
"""

import argparse
import json
import pandas as pd
import re
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any
import os
import sys

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
    # Use explicit mapping to match other evaluation files
    mapping = {
        10: 1, 11: 2, 12: 3, 13: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9,
        20: 10, 21: 12, 22: 11, 23: 13, 24: 14, 25: 15, 26: 16, 27: 17, 28: 18, 29: 19
    }
    
    r_number = int(r_label)
    req_number = mapping.get(r_number, -1)
    
    return str(req_number) if req_number > 0 else r_label

def extract_paragraphs_from_dpa(dpa_segments: pd.DataFrame) -> Dict[str, List[str]]:
    """Extract paragraphs from DPA segments.
    
    This function identifies paragraphs in DPA segments by looking for headings
    and paragraph breaks.
    
    Args:
        dpa_segments: DataFrame containing DPA segments
        
    Returns:
        Dictionary mapping paragraph IDs to lists of segment IDs that make up each paragraph
    """
    paragraphs = {}
    current_paragraph = []
    current_paragraph_id = 0
    
    # Sort segments by their position in the document
    sorted_segments = dpa_segments.sort_values('ID')
    
    for _, row in sorted_segments.iterrows():
        segment_id = str(row['ID'])
        segment_text = row['Sentence']
        
        # Check if this segment starts a new paragraph
        # (Heuristic: if it starts with a number or has special formatting)
        is_new_paragraph = (
            re.match(r'^\d+\.', segment_text) or 
            re.match(r'^[A-Z][A-Z\s]+:', segment_text) or
            len(segment_text.strip()) < 20 and segment_text.strip().endswith(':')
        )
        
        if is_new_paragraph or not current_paragraph:
            # Save the previous paragraph if it exists
            if current_paragraph:
                paragraphs[str(current_paragraph_id)] = current_paragraph
                current_paragraph_id += 1
                
            # Start a new paragraph
            current_paragraph = [segment_id]
        else:
            # Add to current paragraph
            current_paragraph.append(segment_id)
    
    # Save the last paragraph
    if current_paragraph:
        paragraphs[str(current_paragraph_id)] = current_paragraph
    
    return paragraphs

def parse_deolingo_results(deolingo_results: str) -> Dict[Tuple[str, str], str]:
    """Parse deolingo results file to extract statuses.
    
    Args:
        deolingo_results: Content of the deolingo results file
        
    Returns:
        Dictionary mapping (req_id, segment_id) to status
    """
    results = {}
    sections = deolingo_results.split('--------------------------------------------------')
    
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
        results[(req_id, segment_id)] = status
    
    return results

def get_segment_ground_truth(dpa_segments: pd.DataFrame, req_ids: List[str]) -> Dict[str, Set[str]]:
    """Get ground truth requirements for each segment.
    
    Uses individual requirement columns (Requirement-1, Requirement-2, Requirement-3) 
    instead of the consolidated target column for more lenient evaluation.
    
    Args:
        dpa_segments: DataFrame containing DPA segments
        req_ids: List of requirement IDs to evaluate
        
    Returns:
        Dictionary mapping segment IDs to sets of requirement IDs that they satisfy
    """
    segment_to_requirements = defaultdict(set)
    
    for _, row in dpa_segments.iterrows():
        segment_id = str(row['ID'])
        
        # Check all three individual requirement columns
        for col in ['Requirement-1', 'Requirement-2', 'Requirement-3']:
            if col in row and pd.notna(row[col]) and row[col] != 'other':
                # Extract R-label (e.g., 'R10' -> '10')
                req_match = re.match(r'R(\d+)', str(row[col]))
                if req_match:
                    req_id = req_match.group(1)
                    if req_id in req_ids:
                        segment_to_requirements[segment_id].add(req_id)
    
    return segment_to_requirements

def evaluate_binary_classification(
    deolingo_results: Dict[Tuple[str, str], str],
    paragraphs: Dict[str, List[str]],
    segment_to_ground_truth: Dict[str, Set[str]],
    req_ids: List[str]
) -> Dict[str, Any]:
    """Evaluate binary classification metrics for each requirement at the segment level.
    
    For each requirement, evaluate whether individual segments are correctly classified as 
    satisfying that requirement or not.
    
    Args:
        deolingo_results: Parsed deolingo results
        paragraphs: Dictionary mapping paragraph IDs to segment IDs
        segment_to_ground_truth: Dictionary mapping segment IDs to sets of requirements they satisfy
        req_ids: List of requirement IDs to evaluate
        
    Returns:
        Dictionary containing binary classification metrics at segment level
    """
    # Dictionary to store binary classification metrics for each requirement
    binary_metrics = {}
    
    # Get a list of all segments across all paragraphs
    all_segments = []
    for segments in paragraphs.values():
        all_segments.extend(segments)
    all_segments = list(set(all_segments))  # Remove duplicates if any
    
    # Calculate per-requirement metrics
    req_tp_sum, req_fp_sum, req_tn_sum, req_fn_sum = 0, 0, 0, 0
    
    for req_id in req_ids:
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for segment_id in all_segments:
            # Ground truth: Does this segment satisfy this requirement?
            ground_truth = req_id in segment_to_ground_truth.get(segment_id, set())
            
            # Prediction: Did we predict this segment satisfies this requirement?
            prediction = ((req_id, segment_id) in deolingo_results and 
                          deolingo_results[(req_id, segment_id)] == "satisfied")
            
            if ground_truth and prediction:
                tp += 1
                req_tp_sum += 1
            elif ground_truth and not prediction:
                fn += 1
                req_fn_sum += 1
            elif not ground_truth and prediction:
                fp += 1
                req_fp_sum += 1
            else:  # not ground_truth and not prediction
                tn += 1
                req_tn_sum += 1
        
        # Calculate metrics
        req_metrics = calculate_metrics(tp, fp, tn, fn)
        
        # Map requirement ID to human-readable number (R10->1, etc.)
        req_number = map_r_label_to_req_number(req_id)
        
        # Store metrics for this requirement
        binary_metrics[req_id] = {
            "requirement_number": req_number,
            "metrics": req_metrics,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }
        }
    
    # Calculate average metrics across requirements
    num_reqs = len(req_ids)
    avg_req_metrics = calculate_metrics(
        req_tp_sum/num_reqs if num_reqs > 0 else 0,
        req_fp_sum/num_reqs if num_reqs > 0 else 0,
        req_tn_sum/num_reqs if num_reqs > 0 else 0,
        req_fn_sum/num_reqs if num_reqs > 0 else 0
    )
    
    # Now calculate segment-level metrics (treating each segment as a single unit)
    segment_tp, segment_fp, segment_tn, segment_fn = 0, 0, 0, 0
    
    for segment_id in all_segments:
        # Get the ground truth set of requirements for this segment
        gt_reqs = segment_to_ground_truth.get(segment_id, set())
        
        # Get the predicted set of requirements for this segment
        pred_reqs = set()
        for req_id in req_ids:
            if ((req_id, segment_id) in deolingo_results and 
                deolingo_results[(req_id, segment_id)] == "satisfied"):
                pred_reqs.add(req_id)
        
        # Determine the status of this segment
        if gt_reqs and pred_reqs:  # Has ground truth and predictions
            # If any requirements overlap, count as TP
            if gt_reqs & pred_reqs:
                segment_tp += 1
            else:
                segment_fp += 1
        elif gt_reqs and not pred_reqs:  # Has ground truth but no predictions
            segment_fn += 1
        elif not gt_reqs and pred_reqs:  # No ground truth but has predictions
            segment_fp += 1
        else:  # No ground truth and no predictions
            segment_tn += 1
    
    segment_metrics = calculate_metrics(segment_tp, segment_fp, segment_tn, segment_fn)
    
    return {
        "binary_classification": {
            "level": "segment",
            "segment_metrics": {
                "metrics": segment_metrics,
                "confusion_matrix": {
                    "true_positives": segment_tp,
                    "false_positives": segment_fp,
                    "true_negatives": segment_tn,
                    "false_negatives": segment_fn
                },
                "total_segments": len(all_segments)
            },
            "requirement_metrics": {
                "average": avg_req_metrics,
                "per_requirement": binary_metrics
            }
        }
    }

def evaluate_multiclass_classification(
    deolingo_results: Dict[Tuple[str, str], str],
    paragraphs: Dict[str, List[str]],
    segment_to_ground_truth: Dict[str, Set[str]],
    req_ids: List[str]
) -> Dict[str, Any]:
    """Evaluate multi-class classification metrics where each requirement is a class.
    
    Args:
        deolingo_results: Parsed deolingo results
        paragraphs: Dictionary mapping paragraph IDs to segment IDs
        segment_to_ground_truth: Dictionary mapping segment IDs to sets of requirements they satisfy
        req_ids: List of requirement IDs to evaluate
        
    Returns:
        Dictionary containing multi-class classification metrics
    """
    # Create a mapping from paragraph ID to ground truth requirements
    para_to_ground_truth = defaultdict(set)
    for para_id, segments in paragraphs.items():
        for segment_id in segments:
            if segment_id in segment_to_ground_truth:
                para_to_ground_truth[para_id].update(segment_to_ground_truth[segment_id])
    
    # Create a mapping from paragraph ID to predicted requirements
    para_to_predicted = defaultdict(set)
    for para_id, segments in paragraphs.items():
        for segment_id in segments:
            for req_id in req_ids:
                if (req_id, segment_id) in deolingo_results and deolingo_results[(req_id, segment_id)] == "satisfied":
                    para_to_predicted[para_id].add(req_id)
    
    # Count correct and incorrect predictions
    total_paragraphs = len(paragraphs)
    correct_predictions = 0
    
    # Count predictions per paragraph
    para_results = {}
    
    for para_id in paragraphs:
        ground_truth = para_to_ground_truth.get(para_id, set())
        predicted = para_to_predicted.get(para_id, set())
        
        # If the sets of requirements match exactly, it's a correct prediction
        # For paragraphs with no requirements, both sets should be empty
        is_correct = ground_truth == predicted
        
        # If we predicted at least one correct requirement and didn't miss any,
        # consider it partially correct (for softer evaluation)
        predicted_some_correct = bool(ground_truth & predicted)
        missed_none = ground_truth.issubset(predicted) if ground_truth else True
        is_partially_correct = predicted_some_correct and missed_none
        
        if is_correct:
            correct_predictions += 1
        
        para_results[para_id] = {
            "ground_truth": sorted(list(ground_truth)),
            "predicted": sorted(list(predicted)),
            "exact_match": is_correct,
            "partial_match": is_partially_correct,
            "num_correct": len(ground_truth & predicted),
            "num_incorrect": len(predicted - ground_truth),
            "num_missed": len(ground_truth - predicted)
        }
    
    # Calculate accuracy
    accuracy = correct_predictions / total_paragraphs if total_paragraphs > 0 else 0
    
    # Count partial matches
    partial_matches = sum(1 for result in para_results.values() if result["partial_match"])
    partial_accuracy = partial_matches / total_paragraphs if total_paragraphs > 0 else 0
    
    return {
        "multiclass_classification": {
            "accuracy": round(accuracy, 3),
            "partial_accuracy": round(partial_accuracy, 3),
            "total_paragraphs": total_paragraphs,
            "correct_predictions": correct_predictions,
            "partial_matches": partial_matches,
            "paragraph_results": para_results
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate paragraph-level metrics with binary and multi-class classification")
    parser.add_argument("--results", type=str, required=True, help="Path to deolingo results file")
    parser.add_argument("--dpa", type=str, default="data/test_set.csv", help="Path to DPA segments CSV file")
    parser.add_argument("--evaluation", type=str, required=True, help="Path to existing evaluation results JSON")
    parser.add_argument("--output", type=str, required=True, help="Output file for paragraph metrics")
    parser.add_argument("--target_dpa", type=str, default="Online 124", help="Target DPA to evaluate")
    
    args = parser.parse_args()
    
    # Load results file
    with open(args.results, 'r') as f:
        deolingo_content = f.read()
    
    # Load DPA segments
    df = pd.read_csv(args.dpa)
    
    # Filter for target DPA
    dpa_segments = df[df['DPA'] == args.target_dpa]
    
    if dpa_segments.empty:
        print(f"Error: No segments found for DPA '{args.target_dpa}'")
        sys.exit(1)
    
    # Load existing evaluation results for ground truth
    with open(args.evaluation, 'r') as f:
        existing_eval = json.load(f)
    
    # Get the set of requirement IDs from existing evaluation
    req_ids = list(existing_eval.get('requirements_evaluated', []))
    if not req_ids:
        print("No requirement IDs found in evaluation results")
        sys.exit(1)
    
    print(f"Evaluating requirements: {', '.join(req_ids)}")
    
    # Identify paragraphs in the DPA
    paragraphs = extract_paragraphs_from_dpa(dpa_segments)
    print(f"Identified {len(paragraphs)} paragraphs in {args.target_dpa}")
    
    # Parse deolingo results
    parsed_results = parse_deolingo_results(deolingo_content)
    
    # Create mapping of segments to their ground truth requirements
    segment_to_ground_truth = get_segment_ground_truth(dpa_segments, req_ids)
    
    # Evaluate binary classification metrics
    binary_metrics = evaluate_binary_classification(
        parsed_results, paragraphs, segment_to_ground_truth, req_ids
    )
    
    # Evaluate multi-class classification metrics
    multiclass_metrics = evaluate_multiclass_classification(
        parsed_results, paragraphs, segment_to_ground_truth, req_ids
    )
    
    # Combine with existing metrics
    combined_metrics = {
        **existing_eval,
        **binary_metrics,
        **multiclass_metrics
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    
    # Print summary of results
    binary_segment_metrics = binary_metrics['binary_classification']['segment_metrics']['metrics']
    binary_segment_cm = binary_metrics['binary_classification']['segment_metrics']['confusion_matrix']
    binary_req_metrics = binary_metrics['binary_classification']['requirement_metrics']['average']
    
    print("\nBinary Classification Metrics (Segment Level):")
    print(f"Total segments evaluated: {binary_metrics['binary_classification']['segment_metrics']['total_segments']}")
    print(f"Confusion Matrix: TP={binary_segment_cm['true_positives']}, FP={binary_segment_cm['false_positives']}, " 
          f"TN={binary_segment_cm['true_negatives']}, FN={binary_segment_cm['false_negatives']}")
    for metric, value in binary_segment_metrics.items():
        print(f"  {metric.capitalize()}: {value}")
    
    print("\nAverage Per-Requirement Metrics:")
    for metric, value in binary_req_metrics.items():
        print(f"  {metric.capitalize()}: {value}")
    
    print("\nMulti-class Classification Metrics (Paragraph Level):")
    multiclass_results = multiclass_metrics['multiclass_classification']
    print(f"  Accuracy (exact match): {multiclass_results['accuracy']}")
    print(f"  Accuracy (partial match): {multiclass_results['partial_accuracy']}")
    print(f"  Correct Predictions: {multiclass_results['correct_predictions']} of {multiclass_results['total_paragraphs']}")
    
    print(f"\nDetailed metrics saved to: {args.output}")

if __name__ == "__main__":
    main() 