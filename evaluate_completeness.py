# evaluate_completeness.py
import os
import json
import argparse
import pandas as pd
import re
from collections import defaultdict

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
    # Note: In ground_truth_requirements.txt, R5 is the first requirement, R6 is the second, etc.
    if args.req_ids.lower() != "all":
        req_ids = set(id.strip() for id in args.req_ids.split(","))
        print(f"Evaluating requirements: {', '.join(req_ids)}")
    else:
        # Default to requirements R5-R25 if "all" is specified
        # Note: In the ground truth file, R5 is the first requirement, R6 is the second, etc.
        req_ids = set(str(i) for i in range(5, 26))
        print(f"Evaluating all requirements (R5-R25)")
    
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
    
    # Step 5: Get ground truth from target column for filtered segments
    # Note: In the target column, requirements are labeled as R5, R6, etc.
    # where R5 is the first requirement from ground_truth_requirements.txt
    all_ground_truth_reqs = set()
    for _, row in dpa_segments.iterrows():
        target = row['target']
        # Skip 'other' labels
        if target and target != 'other':
            # Extract requirement numbers (R5, R6, etc.)
            req_matches = re.findall(r'R(\d+)', target)
            for req in req_matches:
                # Only add if it's in our filtered requirements
                if req in req_ids:
                    all_ground_truth_reqs.add(req)
    
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
        "requirements_details": {}
    }
    
    # Add detailed requirement information
    for req_id in sorted(filtered_required_reqs):
        ground_truth_satisfied = req_id in all_ground_truth_reqs
        predicted_satisfied = req_id in satisfied_reqs
        
        satisfied_segments = []
        if req_id in results:
            satisfied_segments = [seg for seg, status in results[req_id].items() if status == "satisfied"]
        
        evaluation["requirements_details"][req_id] = {
            "ground_truth": ground_truth_satisfied,
            "prediction": predicted_satisfied,
            "agreement": ground_truth_satisfied == predicted_satisfied,
            "satisfied_segments": sorted(satisfied_segments)
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
    
    print("\nRequirement Details:")
    for req_id in sorted(filtered_required_reqs):
        detail = evaluation["requirements_details"][req_id]
        print(f"\nRequirement R{req_id}:")
        print(f"  Ground Truth: {'Satisfied' if detail['ground_truth'] else 'Not Satisfied'}")
        print(f"  Prediction: {'Satisfied' if detail['prediction'] else 'Not Satisfied'}")
        print(f"  Match: {'Yes' if detail['agreement'] else 'No'}")
        if detail['satisfied_segments']:
            print(f"  Satisfied in segments: {', '.join(detail['satisfied_segments'])}")
        else:
            print("  Not satisfied in any segment")

if __name__ == "__main__":
    main()