# evaluate_results.py
import os
import re
import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def parse_deolingo_results(results_file):
    """Parse the deolingo results file to extract DPA completeness classification."""
    # Initialize structure to hold results by DPA
    dpa_results = {}
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Split the content by DPA-requirement pairs
    pattern = r'Processing DPA ([^,]+), Requirement (\d+)\.\.\..*?-{10,}'
    blocks = re.findall(pattern, content, re.DOTALL)
    
    # Group results by DPA
    for dpa_id, req_id in blocks:
        # Find the corresponding block
        block_start = content.find(f"Processing DPA {dpa_id}, Requirement {req_id}...")
        block_end = content.find("--------------------------------------------------", block_start)
        block_text = content[block_start:block_end]
        
        # Initialize the DPA entry if needed
        if dpa_id not in dpa_results:
            dpa_results[dpa_id] = {'requirements': {}}
        
        # Store requirement result
        if "satisfies(req" in block_text:
            dpa_results[dpa_id]['requirements'][req_id] = "satisfies"
        elif "violates(req" in block_text:
            dpa_results[dpa_id]['requirements'][req_id] = "violates"
        else:
            dpa_results[dpa_id]['requirements'][req_id] = "not_mentioned"
    
    # Determine completeness for each DPA based on requirements 7-24
    for dpa_id in dpa_results:
        reqs = dpa_results[dpa_id]['requirements']
        
        # Check if all requirements 7-24 are satisfied
        req_range = [str(i) for i in range(7, 25)]
        all_satisfied = all(
            req in reqs and reqs[req] == "satisfies" 
            for req in req_range
        )
        
        # Set the completeness prediction
        dpa_results[dpa_id]['predicted_complete'] = all_satisfied
    
    return dpa_results

def compute_ground_truth(df):
    """Compute ground truth completeness for all DPAs."""
    # Get unique DPAs
    dpa_ids = df['DPA'].unique()
    
    # Required requirements range (R7-R24)
    req_range = [f"R{i}" for i in range(7, 25)]
    
    # Calculate ground truth for each DPA
    ground_truth = {}
    
    for dpa_id in dpa_ids:
        # Filter for this DPA
        dpa_df = df[df['DPA'] == dpa_id]
        
        # Get covered requirements
        covered_reqs = set()
        for _, row in dpa_df.iterrows():
            for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
                if col in row and pd.notna(row[col]) and row[col] in req_range:
                    covered_reqs.add(row[col])
        
        # DPA is complete if all required requirements are covered
        is_complete = all(req in covered_reqs for req in req_range)
        
        # Store the result
        ground_truth[dpa_id] = {
            'is_complete': is_complete,
            'covered_reqs': covered_reqs,
            'missing_reqs': set(req for req in req_range if req not in covered_reqs)
        }
    
    return ground_truth

def evaluate_binary_classification(predictions, ground_truth):
    """
    Evaluate the binary classification problem of DPA completeness.
    
    Args:
        predictions: Dictionary mapping DPA IDs to predicted completeness
        ground_truth: Dictionary mapping DPA IDs to actual completeness
        
    Returns:
        Dictionary with precision, recall, F1, and accuracy
    """
    # Extract predictions and actual values
    y_true = []
    y_pred = []
    
    for dpa_id in ground_truth:
        if dpa_id in predictions:
            y_true.append(int(ground_truth[dpa_id]['is_complete']))
            y_pred.append(int(predictions[dpa_id]['predicted_complete']))
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'y_true': y_true,
        'y_pred': y_pred
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate deolingo results as binary classification")
    parser.add_argument("--results", type=str, default="deolingo_results.txt",
                        help="Path to deolingo results file")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--output", type=str, default="evaluation_results.txt",
                        help="Output file for evaluation results")
    args = parser.parse_args()
    
    # Parse deolingo results
    print(f"Parsing deolingo results from: {args.results}")
    dpa_results = parse_deolingo_results(args.results)
    
    # Compute ground truth for all DPAs
    print("Computing ground truth from DPA dataset")
    df = pd.read_csv(args.dpa)
    ground_truth = compute_ground_truth(df)
    
    # Filter to only DPAs that were analyzed
    analyzed_dpas = set(dpa_results.keys())
    filtered_ground_truth = {k: v for k, v in ground_truth.items() if k in analyzed_dpas}
    
    # Evaluate binary classification
    print("Evaluating binary classification of DPA completeness")
    metrics = evaluate_binary_classification(dpa_results, filtered_ground_truth)
    
    # Print results
    print("\n====== Binary Classification Results: DPA Completeness ======")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    
    # Print details for each DPA
    print("\n====== DPA-Level Details ======")
    for dpa_id in sorted(dpa_results.keys()):
        if dpa_id in filtered_ground_truth:
            predicted = dpa_results[dpa_id]['predicted_complete']
            actual = filtered_ground_truth[dpa_id]['is_complete']
            result = "CORRECT" if predicted == actual else "INCORRECT"
            
            print(f"DPA {dpa_id}:")
            print(f"  Predicted: {'Complete' if predicted else 'Incomplete'}")
            print(f"  Actual: {'Complete' if actual else 'Incomplete'}")
            print(f"  Result: {result}")
            
            if not actual:
                print(f"  Missing requirements: {sorted(filtered_ground_truth[dpa_id]['missing_reqs'])}")
            
            print()
    
    # Save results to file
    with open(args.output, 'w') as f:
        f.write("====== Binary Classification Results: DPA Completeness ======\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"True Positives: {metrics['true_positives']}\n")
        f.write(f"False Positives: {metrics['false_positives']}\n")
        f.write(f"False Negatives: {metrics['false_negatives']}\n")
        f.write(f"True Negatives: {metrics['true_negatives']}\n\n")
        
        f.write("====== DPA-Level Details ======\n")
        for dpa_id in sorted(dpa_results.keys()):
            if dpa_id in filtered_ground_truth:
                predicted = dpa_results[dpa_id]['predicted_complete']
                actual = filtered_ground_truth[dpa_id]['is_complete']
                result = "CORRECT" if predicted == actual else "INCORRECT"
                
                f.write(f"DPA {dpa_id}:\n")
                f.write(f"  Predicted: {'Complete' if predicted else 'Incomplete'}\n")
                f.write(f"  Actual: {'Complete' if actual else 'Incomplete'}\n")
                f.write(f"  Result: {result}\n")
                
                if not actual:
                    f.write(f"  Missing requirements: {sorted(filtered_ground_truth[dpa_id]['missing_reqs'])}\n")
                
                f.write("\n")
    
    print(f"Evaluation results saved to: {args.output}")

if __name__ == "__main__":
    main()