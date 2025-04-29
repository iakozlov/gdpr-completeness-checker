# main.py
import os
import argparse
import pandas as pd
from translate import translate_requirements, translate_dpa_segments
from generate_lp import generate_lp_files
from evaluate import evaluate_results

def main():
    parser = argparse.ArgumentParser(description="DPA Compliance Checker")
    parser.add_argument("--requirements", type=str, default="data/requirements/ground_truth_requirements.txt",
                        help="Path to requirements file")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments file")
    parser.add_argument("--model", type=str, help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Step 1: Translate all requirements
    requirements = translate_requirements(args.requirements, args.model)

    # Step 2: Load DPA segments and group by DPA
    df = pd.read_csv(args.dpa)
    dpa_groups = df.groupby("DPA")

    # Step 3: Translate DPA segments
    dpa_translations = translate_dpa_segments(dpa_groups, args.model)

    # Step 4: Generate .lp files for each requirement-DPA pair
    generate_lp_files(requirements, dpa_translations, args.output)

    # Step 5: Generate bash script to run deolingo
    with open("run_deolingo.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# This script runs deolingo on all .lp files and captures results\n")
        f.write("output_file=\"evaluation_results.txt\"\n")
        f.write("echo \"\" > $output_file\n\n")
        
        f.write("# Process all .lp files sequentially\n")
        f.write("for lp_file in $(find results -name \"*.lp\"); do\n")
        f.write("    dpa_id=$(basename $(dirname $lp_file) | sed 's/dpa_//')\n")
        f.write("    req_id=$(basename $lp_file .lp | sed 's/req_//')\n")
        f.write("    \n")
        f.write("    echo \"Processing DPA $dpa_id, Requirement $req_id...\" | tee -a $output_file\n")
        f.write("    python -m deolingo $lp_file | tee -a $output_file\n")
        f.write("    echo \"--------------------------------------------------\" | tee -a $output_file\n")
        f.write("done\n\n")
        
        f.write("echo \"All processing complete. Results saved in $output_file\"\n")
        
    # Make the script executable
    os.chmod("run_deolingo.sh", 0o755)
    
    print("\nDPA compliance checking setup complete.")
    print("To run the deolingo solver on all .lp files:")
    print("  ./run_deolingo.sh")
    print("\nAfter running the solver, evaluate the results with:")
    print(f"  python -c \"import pandas as pd; from evaluate import evaluate_results; evaluate_results(pd.read_csv('{args.dpa}'), '{args.output}')\"")

if __name__ == "__main__":
    main()