# translate_dpa_semantic.py
import os
import json
import argparse
import pandas as pd
import re
from tqdm import tqdm
from models.llm import LlamaModel
from config.llm_config import LlamaConfig

def fix_predicate_name(predicate):
    """Fix predicate names to be valid ASP syntax."""
    predicate = predicate.replace('-', '_')
    predicate = predicate.replace(' ', '_')
    predicate = re.sub(r'[^a-zA-Z0-9_()]', '_', predicate)
    predicate = re.sub(r'_+', '_', predicate)
    predicate = predicate.strip('_')
    
    if predicate and predicate[0].isdigit():
        predicate = 'action_' + predicate
    
    return predicate

def extract_deontic_statements(dpa_segment, llm_model, segment_id):
    """
    Extract deontic logic statements from a DPA segment.
    """
    system_prompt = """
You are a specialized legal analyzer that extracts deontic statements from legal text. 
For each segment of a Data Processing Agreement (DPA), identify the obligations, permissions, 
and prohibitions it contains.

Format your response as deontic logic in Answer Set Programming (ASP) syntax:
1. Use &obligatory{action} for obligations (MUST)
2. Use &permitted{action} for permissions (MAY)
3. Use &forbidden{action} for prohibitions (MUST NOT)
4. Use rules with :- for implications
5. In rule bodies, use commas for conjunctions, NOT the & symbol
6. Ensure all predicates use underscores instead of spaces or hyphens
7. Never put periods inside predicate names

Examples:
&obligatory{obtain_authorization(processor, controller)} :- engage_sub_processor(processor).
&forbidden{engage_sub_processor(processor)} :- not authorization(controller).
&permitted{access_data(controller)} :- data_processing_done(processor).
&permitted{retrieve_data} :- use_controls.
"""

    user_prompt = f"""
Extract deontic statements from this DPA segment (ID: {segment_id}):

"{dpa_segment}"

Provide ONLY the deontic logic statements in ASP format. Examples of valid format:
&obligatory{{action_name}} :- condition.
&permitted{{action_name}} :- not &forbidden{{action_name}}.
&forbidden{{action_name}} :- not authorized.

IMPORTANT:
1. Replace all hyphens with underscores in action names
2. Replace all spaces with underscores  
3. Each statement ends with a period
4. Use only alphanumeric characters and underscores in predicate names
5. Use commas (,) for conjunctions in rule bodies, NOT the & symbol
6. Do not put periods inside predicate names
"""
    
    # Get response from LLM
    response = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
    
    # Extract statements (lines with deontic operators)
    statements = []
    for line in response.split('\n'):
        line = line.strip()
        
        # Check if line contains deontic operators
        if any(op in line for op in ['&obligatory', '&permitted', '&forbidden']):
            # Fix syntax issues
            line = line.replace(' & ', ', ')  # Replace & with comma in conditions
            
            # Ensure the line ends with a period
            if not line.endswith('.'):
                line += '.'
            statements.append(line)
    
    return statements

def translate_dpa_segments(dpa_file, model_path, output_file, target_dpa=None, max_segments=10):
    """
    Translate DPA segments to deontic logic representations.
    """
    # Initialize LLM
    print(f"Initializing LLM with model: {model_path}")
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    print("LLM initialized successfully")
    
    # Read CSV file
    print(f"Reading DPA segments from: {dpa_file}")
    df = pd.read_csv(dpa_file)
    
    # Filter to target DPA or select first one
    if target_dpa:
        filtered_df = df[df["DPA"] == target_dpa]
        if filtered_df.empty:
            print(f"Error: DPA '{target_dpa}' not found in file")
            return
    else:
        # Get unique DPAs and select the first one if no target specified
        unique_dpas = df["DPA"].unique()
        if len(unique_dpas) == 0:
            print("Error: No DPAs found in file")
            return
        target_dpa = unique_dpas[0]
        filtered_df = df[df["DPA"] == target_dpa]
    
    # Limit to max_segments
    if max_segments > 0:
        filtered_df = filtered_df.head(max_segments)
    
    print(f"Processing {len(filtered_df)} segments for DPA: {target_dpa}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process segments
    dpa_data = {
        "dpa_id": target_dpa,
        "segments": []
    }
    
    print("Extracting deontic statements from DPA segments...")
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        segment_id = row["ID"]
        segment_text = row["Sentence"]
        
        # Find related requirements if available
        related_reqs = []
        for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
            if col in row and pd.notna(row[col]) and row[col]:
                related_reqs.append(row[col])
        
        # Extract deontic statements from this segment
        deontic_statements = extract_deontic_statements(segment_text, llm_model, segment_id)
        
        # Store segment data
        dpa_data["segments"].append({
            "id": str(segment_id),
            "text": segment_text,
            "deontic_statements": deontic_statements,
            "requirements": related_reqs
        })
    
    # Save to JSON file
    print(f"Saving symbolic representations to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(dpa_data, f, indent=2)
    
    print("DPA segment translation complete!")
    return dpa_data

def main():
    parser = argparse.ArgumentParser(description="Translate DPA segments to deontic logic representation")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="semantic_results/dpa_deontic.json",
                        help="Output JSON file path")
    parser.add_argument("--target", type=str, default="Online 1",
                        help="Target DPA to process (default: Online 1)")
    parser.add_argument("--max_segments", type=int, default=10,
                        help="Maximum number of segments to process (default: 10, use 0 for all)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Translate DPA segments
    translate_dpa_segments(args.dpa, args.model, args.output, args.target, args.max_segments)

if __name__ == "__main__":
    main()