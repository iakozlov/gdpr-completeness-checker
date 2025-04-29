# evaluate_results.py
import os
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def parse_deolingo_results(results_file, target_dpa="Online_1"):
    """
    Parse the deolingo results file to extract DPA completeness for a specific DPA.
    
    Args:
        results_file: Path to the deolingo results file
        target_dpa: The specific DPA to evaluate (default: "Online_1")
        
    Returns:
        Dictionary with the results for the target DPA
    """
    # Initialize structure to hold results
    dpa_results = {}
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Split the content by DPA-requirement pairs
    pattern = r'Processing DPA ([^,]+), Requirement (\d+)\.\.\..*?-{10,}'
    blocks = re.findall(pattern, content, re.DOTALL)
    
    # Process only the target DPA
    target_blocks = [(dpa_id, req_id) for dpa_id, req_id in blocks if dpa_id == target_dpa]
    
    if not target_blocks:
        print(f"Warning: No results found for DPA {target_dpa}")
        return {}
    
    # Initialize the DPA entry
    dpa_results[target_dpa] = {'requirements': {}}
    
    # Process each requirement for the target DPA
    for dpa_id, req_id in target_blocks:
        # Find the corresponding block
        block_start = content.find(f"Processing DPA {dpa_id}, Requirement {req_id}...")
        block_end = content.find("--------------------------------------------------", block_start)
        if block_end == -1:  # If no end marker found, go to the end of content
            block_end = len(content)
        block_text = content[block_start:block_end]
        
        # Store requirement result
        if "satisfies(req" in block_text:
            dpa_results[dpa_id]['requirements'][req_id] = "satisfies"
        elif "violates(req" in block_text:
            dpa_results[dpa_id]['requirements'][req_id] = "violates"
        elif "SATISFIABLE" in block_text:
            # If we see SATISFIABLE but no explicit satisfies/violates,
            # look for not_mentioned
            if "not_mentioned(req" in block_text:
                dpa_results[dpa_id]['requirements'][req_id] = "not_mentioned"
            else:
                # Default to not_mentioned if no explicit result
                dpa_results[dpa_id]['requirements'][req_id] = "not_mentioned"
        else:
            # Default to not_mentioned if we can't determine
            dpa_results[dpa_id]['requirements'][req_id] = "not_mentioned"
    
    # Determine completeness based on requirements 7-24
    for dpa_id in dpa_results:
        reqs = dpa_results[dpa_id]['requirements']
        
        # Check if all requirements 7-24 are satisfied
        req_range = [str(i) for i in range(7, 25)]
        
        # Count how many of the required requirements are satisfied
        satisfied_count = sum(
            1 for req in req_range 
            if req in reqs and reqs[req] == "satisfies"
        )
        
        # Calculate completeness percentage
        completeness_pct = satisfied_count / len(req_range) if req_range else 0
        
        # Store completeness metrics
        dpa_results[dpa_id]['satisfied_count'] = satisfied_count
        dpa_results[dpa_id]['total_required'] = len(req_range)
        dpa_results[dpa_id]['completeness_pct'] = completeness_pct
        
        # A DPA is considered complete if all required requirements are satisfied
        dpa_results[dpa_id]['predicted_complete'] = completeness_pct == 1.0
    
    return dpa_results

def compute_ground_truth(df, target_dpa="Online 1"):
    """
    Compute ground truth completeness for a specific DPA.
    
    Args:
        df: DataFrame with DPA data
        target_dpa: The specific DPA to evaluate
        
    Returns:
        Dictionary with ground truth for the target DPA
    """
    # Filter for the target DPA
    dpa_df = df[df['DPA'] == target_dpa]
    
    if dpa_df.empty:
        print(f"Warning: DPA '{target_dpa}' not found in dataset")
        return {}
    
    # Required requirements range (R7-R24)
    req_range = [f"R{i}" for i in range(7, 25)]
    
    # Calculate ground truth
    ground_truth = {}
    
    # Get covered requirements
    covered_reqs = set()
    for _, row in dpa_df.iterrows():
        for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
            if col in row and pd.notna(row[col]) and row[col] in req_range:
                covered_reqs.add(row[col])
    
    # Count satisfied requirements
    satisfied_count = len(covered_reqs)
    
    # Calculate completeness percentage
    completeness_pct = satisfied_count / len(req_range) if req_range else 0
    
    # DPA is complete if all required requirements are covered
    is_complete = len(covered_reqs) == len(req_range)
    
    # Store the result
    ground_truth[target_dpa] = {
        'is_complete': is_complete,
        'covered_reqs': covered_reqs,
        'missing_reqs': set(req for req in req_range if req not in covered_reqs),
        'satisfied_count': satisfied_count,
        'total_required': len(req_range),
        'completeness_pct': completeness_pct
    }
    
    # Map from "Online 1" to "Online_1" if necessary
    if target_dpa == "Online 1" and target_dpa not in ground_truth:
        ground_truth["Online_1"] = ground_truth[target_dpa]
        del ground_truth[target_dpa]
    elif target_dpa == "Online_1" and target_dpa not in ground_truth:
        ground_truth["Online_1"] = ground_truth["Online 1"]
        del ground_truth["Online 1"]
    
    return ground_truth

def evaluate_dpa_completeness(predictions, ground_truth, target_dpa):
    """
    Evaluate the completeness prediction for a specific DPA.
    
    Args:
        predictions: Dictionary with DPA predictions
        ground_truth: Dictionary with DPA ground truth
        target_dpa: The specific DPA to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Check if target DPA exists in both predictions and ground truth
    target_dpa_underscore = target_dpa.replace(' ', '_')
    
    # Try both formats (with space and with underscore)
    prediction_key = target_dpa if target_dpa in predictions else target_dpa_underscore
    truth_key = target_dpa if target_dpa in ground_truth else target_dpa_underscore
    
    if prediction_key not in predictions or truth_key not in ground_truth:
        print(f"Warning: DPA '{target_dpa}' not found in both predictions and ground truth")
        return {
            'correct': False,
            'predicted_complete': False,
            'actual_complete': False,
            'predicted_satisfied_count': 0,
            'actual_covered_count': 0,
            'total_required': 18  # R7-R24 = 18 requirements
        }
    
    # Get the prediction and ground truth
    prediction = predictions[prediction_key]['predicted_complete']
    actual = ground_truth[truth_key]['is_complete']
    
    # Calculate metrics
    return {
        'correct': prediction == actual,
        'predicted_complete': prediction,
        'actual_complete': actual,
        'predicted_satisfied_count': predictions[prediction_key]['satisfied_count'],
        'actual_covered_count': len(ground_truth[truth_key]['covered_reqs']),
        'total_required': predictions[prediction_key]['total_required'],
        'missing_reqs': ground_truth[truth_key]['missing_reqs'] if not actual else set()
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate deolingo results for DPA Online 1")
    parser.add_argument("--results", type=str, default="deolingo_results.txt",
                        help="Path to deolingo results file")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--output", type=str, default="evaluation_results.txt",
                        help="Output file for evaluation results")
    parser.add_argument("--target", type=str, default="Online 1",
                        help="Target DPA to evaluate (default: Online 1)")
    args = parser.parse_args()
    
    target_dpa = args.target
    
    # Parse deolingo results for the target DPA
    print(f"Parsing deolingo results for DPA '{target_dpa}' from: {args.results}")
    dpa_results = parse_deolingo_results(args.results, target_dpa.replace(' ', '_'))
    
    if not dpa_results:
        print(f"No results found for DPA '{target_dpa}'. Please check if this DPA exists in the results.")
        return
    
    # Compute ground truth for the target DPA
    print(f"Computing ground truth for DPA '{target_dpa}' from dataset")
    df = pd.read_csv(args.dpa)
    ground_truth = compute_ground_truth(df, target_dpa)
    
    if not ground_truth:
        print(f"No ground truth found for DPA '{target_dpa}'. Please check if this DPA exists in the dataset.")
        return
    
    # Evaluate completeness prediction
    print(f"Evaluating completeness prediction for DPA '{target_dpa}'")
    evaluation = evaluate_dpa_completeness(dpa_results, ground_truth, target_dpa)
    
    # Print results
    print(f"\n====== Evaluation Results for DPA '{target_dpa}' ======")
    print(f"Predicted: {'Complete' if evaluation['predicted_complete'] else 'Incomplete'} ({evaluation['predicted_satisfied_count']}/{evaluation['total_required']} requirements satisfied)")
    print(f"Actual: {'Complete' if evaluation['actual_complete'] else 'Incomplete'} ({evaluation['actual_covered_count']}/{evaluation['total_required']} requirements covered)")
    print(f"Result: {'CORRECT' if evaluation['correct'] else 'INCORRECT'}")
    
    if not evaluation['actual_complete']:
        print(f"Missing requirements: {sorted(evaluation['missing_reqs'])}")
    
    # Print requirement-level details
    target_dpa_underscore = target_dpa.replace(' ', '_')
    prediction_key = target_dpa if target_dpa in dpa_results else target_dpa_underscore
    
    if prediction_key in dpa_results:
        print("\n====== Requirement-Level Details ======")
        req_results = dpa_results[prediction_key]['requirements']
        
        # Focus on the critical range R7-R24
        critical_range = range(7, 25)
        
        for req_id in critical_range:
            req_id_str = str(req_id)
            status = req_results.get(req_id_str, "unknown")
            print(f"Requirement {req_id_str}: {status.upper()}")
    
    # Save results to file
    with open(args.output, 'w') as f:
        f.write(f"====== Evaluation Results for DPA '{target_dpa}' ======\n")
        f.write(f"Predicted: {'Complete' if evaluation['predicted_complete'] else 'Incomplete'} ({evaluation['predicted_satisfied_count']}/{evaluation['total_required']} requirements satisfied)\n")
        f.write(f"Actual: {'Complete' if evaluation['actual_complete'] else 'Incomplete'} ({evaluation['actual_covered_count']}/{evaluation['total_required']} requirements covered)\n")
        f.write(f"Result: {'CORRECT' if evaluation['correct'] else 'INCORRECT'}\n")
        
        if not evaluation['actual_complete']:
            f.write(f"Missing requirements: {sorted(evaluation['missing_reqs'])}\n")
        
        # Add requirement-level details
        if prediction_key in dpa_results:
            f.write("\n====== Requirement-Level Details ======\n")
            req_results = dpa_results[prediction_key]['requirements']
            
            # Focus on the critical range R7-R24
            critical_range = range(7, 25)
            
            for req_id in critical_range:
                req_id_str = str(req_id)
                status = req_results.get(req_id_str, "unknown")
                f.write(f"Requirement {req_id_str}: {status.upper()}\n")
    
    print(f"\nEvaluation results saved to: {args.output}")

if __name__ == "__main__":
    main()