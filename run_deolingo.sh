#!/bin/bash
# run_deolingo.sh

# First check for dependencies
echo "Checking and installing dependencies..."
pip install python-dotenv clingo

# Directory containing LP files
RESULTS_DIR="results"

# Output file for results
OUTPUT_FILE="deolingo_results.txt"

# Clear any previous results
echo "" > $OUTPUT_FILE

# Function to process a single LP file
process_file() {
    local lp_file=$1
    
    # Extract DPA and requirement IDs from file path
    local dpa_id=$(basename $(dirname $lp_file) | sed 's/dpa_//')
    local req_id=$(basename $lp_file .lp | sed 's/req_//')
    
    echo "Processing DPA $dpa_id, Requirement $req_id..." | tee -a $OUTPUT_FILE
    
    # Run deolingo on the file
    python -m deolingo $lp_file | tee -a $OUTPUT_FILE
    
    # Add a separator
    echo "--------------------------------------------------" | tee -a $OUTPUT_FILE
}

# Find all LP files and process them
find $RESULTS_DIR -name "*.lp" | while read lp_file; do
    process_file $lp_file
done

echo "All files processed. Results saved to $OUTPUT_FILE"