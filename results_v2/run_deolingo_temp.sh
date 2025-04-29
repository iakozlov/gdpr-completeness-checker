#!/bin/bash
# Temporary script to run deolingo on all .lp files

OUTPUT_FILE="results_v2/deolingo_results.txt"
echo "" > $OUTPUT_FILE

# Process all .lp files
find results_v2 -name "*.lp" | while read lp_file; do
    dpa_id=$(basename $(dirname $lp_file) | sed 's/dpa_//')
    req_id=$(basename $lp_file .lp | sed 's/req_//')
    
    echo "Processing DPA $dpa_id, Requirement $req_id..." | tee -a $OUTPUT_FILE
    deolingo $lp_file | tee -a $OUTPUT_FILE
    echo "--------------------------------------------------" | tee -a $OUTPUT_FILE
done

echo "All processing complete. Results saved in $OUTPUT_FILE"
