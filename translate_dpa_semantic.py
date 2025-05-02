# translate_dpa_semantic.py
import os
import json
import argparse
import pandas as pd
import re
from tqdm import tqdm
from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def extract_dpa_actions(dpa_segment, llm_model, segment_id):
    """
    Extract potential actions from a DPA segment.
    
    Args:
        dpa_segment: Text of the DPA segment
        llm_model: Initialized LLM model
        segment_id: ID of the segment for tracking
        
    Returns:
        List of deontic statements representing actions in the DPA segment
    """
    system_prompt = """
You are a specialized legal analyzer that extracts deontic statements from legal text. 
For each segment of a Data Processing Agreement (DPA), identify the obligations, permissions, 
and prohibitions it contains.

Format your response as a list of deontic statements in Answer Set Programming (ASP) syntax:
1. Use dpa_states(id, modality, action) format
2. For modality, use: obligatory, permitted, or forbidden
3. For actions, use descriptive predicate names with processor/controller as arguments
"""

    user_prompt = f"""
Extract deontic statements from this DPA segment (ID: {segment_id}):

"{dpa_segment}"

Provide ONLY the dpa_states() predicates, nothing else:
"""
    
    # Get response from LLM
    response = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
    
    # Extract dpa_states predicates
    predicates = []
    pattern = r'dpa_states\([^)]+\)'
    matches = re.findall(pattern, response)
    if matches:
        predicates = matches
    
    return predicates

def translate_dpa_segments(dpa_file, model_path, output_file, target_dpa=None):
    """
    Translate DPA segments to symbolic representations.
    
    Args:
        dpa_file: Path to DPA segments CSV file
        model_path: Path to LLM model file
        output_file: Output file path for JSON results
        target_dpa: Optional DPA ID to filter (if None, use first DPA found)
    """
    # Initialize LLM and translator
    print(f"Initializing LLM with model: {model_path}")
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    translator = DeonticTranslator()
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
    
    print(f"Processing {len(filtered_df)} segments for DPA: {target_dpa}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process segments
    dpa_data = {
        "dpa_id": target_dpa,
        "segments": {}
    }
    
    print("Extracting actions from DPA segments...")
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        segment_id = row["ID"]
        segment_text = row["Sentence"]
        
        # Find related requirements if available
        related_reqs = []
        for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
            if col in row and pd.notna(row[col]) and row[col]:
                related_reqs.append(row[col])
        
        # Extract actions from this segment
        segment_actions = extract_dpa_actions(segment_text, llm_model, segment_id)
        
        # Store segment data
        dpa_data["segments"][segment_id] = {
            "text": segment_text,
            "actions": segment_actions,
            "requirements": related_reqs
        }
    
    # Save to JSON file
    print(f"Saving symbolic representations to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(dpa_data, f, indent=2)
    
    print("DPA segment translation complete!")

def main():
    parser = argparse.ArgumentParser(description="Translate DPA segments to symbolic representation")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="semantic_results/dpa_symbolic.json",
                        help="Output JSON file path")
    parser.add_argument("--target", type=str, default="Online 1",
                        help="Target DPA to process (default: Online 1)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Translate DPA segments
    translate_dpa_segments(args.dpa, args.model, args.output, args.target)

if __name__ == "__main__":
    main()