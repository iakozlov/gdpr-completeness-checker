# evaluate_individual_results.py
import os
import re
import argparse
import json
import subprocess
from tqdm import tqdm

def run_deolingo_on_file(lp_file_path):
    """Run deolingo on a single LP file and return the result."""
    try:
        result = subprocess.run(
            ["deolingo", lp_file_path],
            capture_output=True, 
            text=True,
            check=False
        )
        
        # Parse result to determine if satisfies, violates, or not_mentioned
        stdout = result.stdout
        
        if "satisfies(req)" in stdout:
            return "satisfies"
        elif "violates(req)" in stdout:
            return "violates"
        elif "not_mentioned(req)" in stdout:
            return "not_mentioned"
        else:
            print(f"Warning: Unexpected output from deolingo for {lp_file_path}")
            print(f"Output: {stdout}")
            return "error"
            
    except Exception as e:
        print(f"Error running deolingo on {lp_file_path}: {e}")
        return "error"

def evaluate_individual_lp_files(lp_dir, output_file):
    """Evaluate individual LP files using deolingo and save results."""
    # Get all LP files
    lp_files = [f for f in os.listdir(lp_dir) if f.endswith(".lp")]
    
    # Sort files for consistent processing
    lp_files.sort()
    
    # Initialize results
    results = {
        "req_id": os.path.basename(lp_dir).replace("req_", ""),
        "total_segments": len(lp_files),
        "segments": {},
        "summary": {
            "satisfies": 0,
            "violates": 0,
            "not_mentioned": 0,
            "error": 0
        }
    }
    
    # Process each LP file
    print(f"Processing {len(lp_files)} LP files in {lp_dir}")
    
    for lp_file in tqdm(lp_files):
        # Extract segment ID
        segment_id = lp_file.replace("segment_", "").replace(".lp", "")
        
        # Full path to LP file
        lp_file_path = os.path.join(lp_dir, lp_file)
        
        # Run deolingo
        print(f"Running Deolingo on Requirement {results['req_id']}, Segment {segment_id}...")
        result = run_deolingo_on_file(lp_file_path)
        
        # Save result
        results["segments"][segment_id] = result
        
        # Update summary counts
        results["summary"][result] += 1
        
        # Add separator
        print("--------------------------------------------------")
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Total segments: {len(lp_files)}")
    print(f"Satisfies: {results['summary']['satisfies']}")
    print(f"Violates: {results['summary']['violates']}")
    print(f"Not mentioned: {results['summary']['not_mentioned']}")
    print(f"Errors: {results['summary']['error']}")
    print(f"Results saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate individual LP files with deolingo")
    parser.add_argument("--lp_dir", type=str, required=True,
                       help="Directory containing LP files")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Check if LP directory exists
    if not os.path.exists(args.lp_dir):
        print(f"Error: LP directory not found: {args.lp_dir}")
        return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Evaluate LP files
    evaluate_individual_lp_files(args.lp_dir, args.output)

if __name__ == "__main__":
    main()