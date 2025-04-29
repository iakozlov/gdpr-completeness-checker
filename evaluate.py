# evaluate.py
import os
import re
import pandas as pd

def parse_deolingo_results(file_path):
    """Parse the output of deolingo to extract satisfies/violates/not_mentioned."""
    results = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Split by processing blocks
        processing_blocks = re.findall(r'Processing DPA (\d+), Requirement (\d+)\.\.\..*?-{10,}', content, re.DOTALL)
        
        for match in processing_blocks:
            dpa_id, req_id = match
            
            # Extract the content for this DPA-requirement pair
            block_start = content.find(f"Processing DPA {dpa_id}, Requirement {req_id}...")
            block_end = content.find("--------------------------------------------------", block_start)
            block_content = content[block_start:block_end]
            
            # Initialize DPA in results if needed
            if dpa_id not in results:
                results[dpa_id] = {}
            
            # Extract result
            if "satisfies(req" in block_content:
                results[dpa_id][req_id] = "satisfies"
            elif "violates(req" in block_content:
                results[dpa_id][req_id] = "violates"
            else:
                results[dpa_id][req_id] = "not_mentioned"
    
    return results

def compute_ground_truth(df):
    """Compute ground truth labels based on requirement coverage."""
    # Extract unique DPAs
    dpas = df["DPA"].unique()
    
    # Initialize ground truth
    ground_truth = {}
    
    # Check each DPA
    for dpa in dpas:
        # Filter rows for this DPA
        dpa_rows = df[df["DPA"] == dpa]
        
        # Check if all requirements R7-R24 are covered
        required_reqs = [f"R{i}" for i in range(7, 25)]
        covered_reqs = []
        
        for _, row in dpa_rows.iterrows():
            for req_col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
                if req_col in row and row[req_col] in required_reqs:
                    covered_reqs.append(row[req_col])
        
        # Remove duplicates
        covered_reqs = list(set(covered_reqs))
        
        # DPA is complete if all required requirements are covered
        is_complete = all(req in covered_reqs for req in required_reqs)
        ground_truth[str(dpa)] = is_complete
    
    return ground_truth

def calculate_metrics(predictions, ground_truth):
    """Calculate precision, recall, and F1 score."""
    tp = 0  # True Positive: predicted complete and actually complete
    fp = 0  # False Positive: predicted complete but actually incomplete
    fn = 0  # False Negative: predicted incomplete but actually complete
    tn = 0  # True Negative: predicted incomplete and actually incomplete
    
    for dpa_id, is_complete in ground_truth.items():
        if dpa_id in predictions:
            # Get requirement IDs that match R7-R24
            req_mapping = {str(i): f"R{i}" for i in range(7, 25)}
            relevant_reqs = [req_id for req_id in predictions[dpa_id].keys() if req_id in req_mapping.keys()]
            
            # DPA is predicted complete if all relevant requirements are satisfied
            pred_complete = all(predictions[dpa_id].get(req_id, "not_mentioned") == "satisfies" 
                              for req_id in relevant_reqs)
            
            if pred_complete and is_complete:
                tp += 1
            elif pred_complete and not is_complete:
                fp += 1
            elif not pred_complete and is_complete:
                fn += 1
            else:  # not pred_complete and not is_complete
                tn += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    }

def evaluate_results(df, output_dir):
    """Evaluate the results and calculate metrics."""
    # Parse deolingo results
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Please run the deolingo solver first using run_deolingo.sh")
        return
    
    results = parse_deolingo_results(results_file)
    
    # Compute ground truth
    ground_truth = compute_ground_truth(df)
    
    # Calculate metrics
    metrics = calculate_metrics(results, ground_truth)
    
    # Print results
    print("\n--- Evaluation Results ---")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"True Positives: {metrics['true_positives']}\n")
        f.write(f"False Positives: {metrics['false_positives']}\n")
        f.write(f"True Negatives: {metrics['true_negatives']}\n")
        f.write(f"False Negatives: {metrics['false_negatives']}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")