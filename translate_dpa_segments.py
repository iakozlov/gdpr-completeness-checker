# translate_dpa_segments.py
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def main():
    parser = argparse.ArgumentParser(description="Translate DPA segments to symbolic representation")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="data/processed/dpa_segments_symbolic.json",
                        help="Output JSON file path")
    parser.add_argument("--filter", type=str, default="Online 1",
                        help="Filter DPA segments by this value in the DPA column")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize LLM and translator
    print(f"Initializing LLM with model: {args.model}")
    llm_config = LlamaConfig(model_path=args.model, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    translator = DeonticTranslator()
    print("LLM initialized successfully")
    
    # Read CSV file
    print(f"Reading DPA segments from: {args.dpa}")
    df = pd.read_csv(args.dpa)
    
    # Filter for the specified DPA
    filtered_df = df[df["DPA"] == args.filter]
    print(f"Found {len(filtered_df)} segments for DPA: {args.filter}")
    
    # Process segments
    dpa_data = {
        "dpa_id": args.filter,
        "segments": {}
    }
    
    print("Translating segments...")
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        segment_id = row["ID"]
        segment_text = row["Sentence"]
        
        # Generate symbolic representation
        llm_output = llm_model.generate_symbolic_representation(segment_text)
        symbolic = translator.translate(llm_output)
        
        # Store in dictionary with metadata
        dpa_data["segments"][segment_text] = {
            "id": segment_id,
            "symbolic": symbolic,
            "requirements": [
                row.get("Requirement-1", ""),
                row.get("Requirement-2", ""),
                row.get("Requirement-3", "")
            ]
        }
    
    # Save to JSON file
    print(f"Saving symbolic representations to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(dpa_data, f, indent=2)
    
    print("DPA segment translation complete!")

if __name__ == "__main__":
    main()