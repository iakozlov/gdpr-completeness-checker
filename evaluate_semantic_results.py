# evaluate_semantic_results.py
import os
import re
import argparse
import pandas as pd
import json

def load_requirement_texts(requirements_file="data/requirements/ground_truth_requirements.txt"):
    """Load requirement texts from ground_truth_requirements.txt file."""
    requirement_texts = {}
    if not os.path.exists(requirements_file):
        print(f"Warning: Requirements file not found at {requirements_file}")
        return requirement_texts
    
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract requirement ID and text (format: "1. The processor shall...")
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if match:
                req_id = match.group(1)
                req_text = match.group(2)
                requirement_texts[req_id] = req_text
    
    return requirement_texts

def parse_deolingo_results(results_file, target_dpa="Online_1"):
    """Parse the deolingo results file to extract DPA completeness for a specific DPA."""
    # Initialize structure to hold results
    dpa_results = {}
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Split the content by DPA-requirement pairs
    pattern = r'Processing DPA ([^,]+), Requirement (\d+)\.\.\..*?-{10,}'
    blocks = re.findall(pattern, content, re.DOTALL)
    
    # Process only the target DPA
    target_blocks = [(dpa_id, req_id) for dpa_id, req_id in blocks if dpa_id.replace('_', ' ') == target_dpa.replace('_', ' ')]
    
    if not target_blocks:
        print(f"Warning: No results found for DPA {target_dpa}")
        return {}
    
    # Initialize the DPA entry
    normalized_dpa_id = target_dpa.replace(' ', '_')
    dpa_results[normalized_dpa_id] = {'requirements': {}}
    
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
            dpa_results[normalized_dpa_id]['requirements'][req_id] = "satisfies"
        elif "violates(req" in block_text:
            dpa_results[normalized_dpa_id]['requirements'][req_id] = "violates"
        elif "SATISFIABLE" in block_text:
            # If we see SATISFIABLE but no explicit satisfies/violates,
            # look for not_mentioned
            if "not_mentioned(req" in block_text:
                dpa_results[normalized_dpa_id]['requirements'][req_id] = "not_mentioned"
            else:
                # Default to not_mentioned if no explicit result
                dpa_results[normalized_dpa_id]['requirements'][req_id] = "not_mentioned"
        else:
            # Default to not_mentioned if we can't determine
            dpa_results[normalized_dpa_id]['requirements'][req_id] = "not_mentioned"
    
    # Calculate summary metrics
    for dpa_id in dpa_results:
        reqs = dpa_results[dpa_id]['requirements']
        
        # Count satisfied requirements
        satisfied_count = sum(1 for status in reqs.values() if status == "satisfies")
        
        # Store metrics
        dpa_results[dpa_id]['satisfied_count'] = satisfied_count
        dpa_results[dpa_id]['total_evaluated'] = len(reqs)
    
    return dpa_results

def compute_ground_truth(df, target_dpa="Online 1"):
    """Compute ground truth completeness for a specific DPA."""
    # Filter for the target DPA
    dpa_df = df[df['DPA'] == target_dpa]
    
    if dpa_df.empty:
        print(f"Warning: DPA '{target_dpa}' not found in dataset")
        return {}
    
    # Required requirements range (R7-R24)
    req_range = [f"R{i}" for i in range(7, 25)]
    
    # Get covered requirements
    covered_reqs = set()
    for _, row in dpa_df.iterrows():
        for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
            if col in row and pd.notna(row[col]) and row[col] in req_range:
                covered_reqs.add(row[col])
    
    # DPA is complete if all required requirements are covered
    is_complete = len(covered_reqs) == len(req_range)
    
    # Store the result
    normalized_dpa_id = target_dpa.replace(' ', '_')
    ground_truth = {
        normalized_dpa_id: {
            'is_complete': is_complete,
            'covered_reqs': covered_reqs,
            'missing_reqs': set(req for req in req_range if req not in covered_reqs),
            'satisfied_count': len(covered_reqs),
            'total_required': len(req_range)
        }
    }
    
    return ground_truth

def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic mapping results for DPA compliance")
    parser.add_argument("--results", type=str, default="semantic_results/deolingo_results.txt",
                        help="Path to deolingo results file")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--requirements_text", type=str, default="data/requirements/ground_truth_requirements.txt",
                        help="Path to requirement texts file")
    parser.add_argument("--output", type=str, default="semantic_results/evaluation_results.txt",
                        help="Output file for evaluation results")
    parser.add_argument("--target", type=str, default="Online 1",
                        help="Target DPA to evaluate (default: Online 1)")
    args = parser.parse_args()
    
    target_dpa = args.target
    
    # Load requirement texts from ground_truth_requirements.txt
    requirement_texts = load_requirement_texts(args.requirements_text)
    
    # Parse deolingo results for the target DPA
    print(f"Parsing deolingo results for DPA '{target_dpa}' from: {args.results}")
    dpa_results = parse_deolingo_results(args.results, target_dpa)
    
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
    
    # Get normalized DPA ID
    normalized_dpa_id = target_dpa.replace(' ', '_')
    
    # Print results
    print(f"\n====== Evaluation Results for DPA '{target_dpa}' ======")
    
    if normalized_dpa_id in dpa_results and normalized_dpa_id in ground_truth:
        predicted_satisfied = dpa_results[normalized_dpa_id]['satisfied_count']
        predicted_total = dpa_results[normalized_dpa_id]['total_evaluated']
        
        actual_satisfied = ground_truth[normalized_dpa_id]['satisfied_count']
        actual_total = ground_truth[normalized_dpa_id]['total_required']
        
        predicted_complete = predicted_satisfied == predicted_total
        actual_complete = ground_truth[normalized_dpa_id]['is_complete']
        
        print(f"Predicted: {'Complete' if predicted_complete else 'Incomplete'} ({predicted_satisfied}/{predicted_total} requirements satisfied)")
        print(f"Actual: {'Complete' if actual_complete else 'Incomplete'} ({actual_satisfied}/{actual_total} requirements covered)")
        print(f"Result: {'CORRECT' if predicted_complete == actual_complete else 'INCORRECT'}")
    
    # Print detailed requirement-level results
    print("\n====== Requirement-Level Details ======")
    print(f"{'Req ID':<8} {'Status':<15} {'Requirement Text'}")
    print("-" * 120)
    
    if normalized_dpa_id in dpa_results:
        req_results = dpa_results[normalized_dpa_id]['requirements']
        
        # Focus on evaluated requirements
        for req_id, status in sorted(req_results.items(), key=lambda x: int(x[0])):
            req_name = f"R{req_id}"
            
            # Get requirement text if available
            req_text = requirement_texts.get(req_id, "Text not found")
            
            # Truncate text if too long
            if len(req_text) > 80:
                req_text = req_text[:77] + "..."
            
            print(f"{req_name:<8} {status.upper():<15} {req_text}")
    
    # Print ground truth coverage comparison
    print("\n====== Ground Truth Coverage Comparison ======")
    if normalized_dpa_id in ground_truth:
        covered_reqs = ground_truth[normalized_dpa_id]['covered_reqs']
        missing_reqs = ground_truth[normalized_dpa_id]['missing_reqs']
        
        print(f"Requirements covered by ground truth: {sorted(covered_reqs, key=lambda x: int(x[1:]))}")
        print(f"Requirements missing from ground truth: {sorted(missing_reqs, key=lambda x: int(x[1:]))}")
        
        if normalized_dpa_id in dpa_results:
            req_results = dpa_results[normalized_dpa_id]['requirements']
            
            # Find false positives and false negatives
            predicted_satisfied_reqs = set(f"R{req_id}" for req_id, status in req_results.items() if status == "satisfies")
            false_positives = predicted_satisfied_reqs - covered_reqs
            false_negatives = covered_reqs - predicted_satisfied_reqs
            
            if false_positives:
                print(f"\nFalse Positives (predicted satisfied but not in ground truth): {sorted(false_positives, key=lambda x: int(x[1:]))}")
            if false_negatives:
                print(f"False Negatives (in ground truth but not predicted as satisfied): {sorted(false_negatives, key=lambda x: int(x[1:]))}")
    
    # Save results to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(f"====== Evaluation Results for DPA '{target_dpa}' ======\n\n")
        
        if normalized_dpa_id in dpa_results and normalized_dpa_id in ground_truth:
            f.write(f"Predicted: {'Complete' if predicted_complete else 'Incomplete'} ({predicted_satisfied}/{predicted_total} requirements satisfied)\n")
            f.write(f"Actual: {'Complete' if actual_complete else 'Incomplete'} ({actual_satisfied}/{actual_total} requirements covered)\n")
            f.write(f"Result: {'CORRECT' if predicted_complete == actual_complete else 'INCORRECT'}\n\n")
        
        f.write("====== Requirement-Level Details ======\n")
        f.write(f"{'Req ID':<8} {'Status':<15} {'Requirement Text'}\n")
        f.write("-" * 120 + "\n")
        
        if normalized_dpa_id in dpa_results:
            req_results = dpa_results[normalized_dpa_id]['requirements']
            
            for req_id, status in sorted(req_results.items(), key=lambda x: int(x[0])):
                req_name = f"R{req_id}"
                req_text = requirement_texts.get(req_id, "Text not found")
                
                f.write(f"{req_name:<8} {status.upper():<15} {req_text}\n")
        
        # Write ground truth comparison to file
        f.write("\n====== Ground Truth Coverage Comparison ======\n")
        if normalized_dpa_id in ground_truth:
            covered_reqs = ground_truth[normalized_dpa_id]['covered_reqs']
            missing_reqs = ground_truth[normalized_dpa_id]['missing_reqs']
            
            f.write(f"Requirements covered by ground truth: {sorted(covered_reqs, key=lambda x: int(x[1:]))}\n")
            f.write(f"Requirements missing from ground truth: {sorted(missing_reqs, key=lambda x: int(x[1:]))}\n")
            
            if normalized_dpa_id in dpa_results:
                req_results = dpa_results[normalized_dpa_id]['requirements']
                
                predicted_satisfied_reqs = set(f"R{req_id}" for req_id, status in req_results.items() if status == "satisfies")
                false_positives = predicted_satisfied_reqs - covered_reqs
                false_negatives = covered_reqs - predicted_satisfied_reqs
                
                if false_positives:
                    f.write(f"\nFalse Positives: {sorted(false_positives, key=lambda x: int(x[1:]))}\n")
                if false_negatives:
                    f.write(f"False Negatives: {sorted(false_negatives, key=lambda x: int(x[1:]))}\n")
    
    print(f"\nEvaluation results saved to: {args.output}")

if __name__ == "__main__":
    main()