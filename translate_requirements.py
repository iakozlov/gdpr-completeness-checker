# translate_requirements.py
import os
import re
import json
import argparse
from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def main():
    parser = argparse.ArgumentParser(description="Translate GDPR requirements to symbolic representation")
    parser.add_argument("--requirements", type=str, default="data/requirements/ground_truth_requirements.txt",
                        help="Path to requirements file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="data/processed/requirements_symbolic.json",
                        help="Output JSON file path")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize LLM and translator
    print(f"Initializing LLM with model: {args.model}")
    llm_config = LlamaConfig(model_path=args.model, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    translator = DeonticTranslator()
    print("LLM initialized successfully")
    
    # Read and parse requirements
    print(f"Reading requirements from: {args.requirements}")
    requirements = {}
    with open(args.requirements, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract requirement ID and text (format: "1. The processor shall...")
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if match:
                req_id = match.group(1)
                req_text = match.group(2)
                
                print(f"Translating requirement {req_id}: {req_text[:50]}...")
                
                # Generate symbolic representation
                llm_output = llm_model.generate_symbolic_representation(req_text)
                symbolic = translator.translate(llm_output)
                
                # Store in dictionary
                requirements[req_text] = symbolic
                print(f"  → Symbolic: {symbolic[:70]}...")
    
    # Save to JSON file
    print(f"Saving {len(requirements)} symbolic representations to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(requirements, f, indent=2)
    
    print("Requirement translation complete!")

if __name__ == "__main__":
    main()