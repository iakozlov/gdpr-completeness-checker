# translate_requirements.py
import os
import re
import json
import argparse
from models.gpt_model import GPTModel
from models.llama_model import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.gpt_config import GPTConfig
from config.llama_config import LlamaConfig

def main():
    parser = argparse.ArgumentParser(description="Translate GDPR requirements to deontic logic")
    parser.add_argument("--requirements", type=str, default="data/requirements/ground_truth_requirements.txt",
                        help="Path to requirements file")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use (gpt-4o or meta-llama/Llama-3.3-70B-Instruct)")
    parser.add_argument("--output", type=str, default="results/gpt4o_experiment/requirements_deontic.json",
                        help="Output JSON file path")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize LLM and translator
    print(f"Initializing model: {args.model}")
    
    # Choose the appropriate model based on the model name
    if args.model.startswith("meta-llama"):
        model_config = LlamaConfig(model=args.model, temperature=0.1)
        llm_model = LlamaModel(model_config)
    else:
        model_config = GPTConfig(model=args.model, temperature=0.1)
        llm_model = GPTModel(model_config)
    
    translator = DeonticTranslator()
    print("Model initialized successfully")
    
    # Read and parse requirements
    print(f"Reading requirements from: {args.requirements}")
    requirements = {}
    
    with open(args.requirements, 'r') as f:
        current_req_id = None
        current_req_text = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a requirement ID line
            req_match = re.match(r'^R(\d+):', line)
            if req_match:
                # Save previous requirement if exists
                if current_req_id is not None:
                    requirements[current_req_id] = {
                        'text': ' '.join(current_req_text),
                        'symbolic': None  # Will be filled later
                    }
                
                # Start new requirement
                current_req_id = req_match.group(1)
                current_req_text = [line]
            else:
                # Add to current requirement text
                current_req_text.append(line)
        
        # Save last requirement
        if current_req_id is not None:
            requirements[current_req_id] = {
                'text': ' '.join(current_req_text),
                'symbolic': None
            }
    
    # Translate each requirement to deontic logic
    print("Translating requirements to deontic logic...")
    for req_id, req_data in requirements.items():
        print(f"Processing requirement {req_id}...")
        try:
            # Generate symbolic representation
            symbolic_repr = llm_model.generate_symbolic_representation(req_data['text'])
            requirements[req_id]['symbolic'] = symbolic_repr
        except Exception as e:
            print(f"Error processing requirement {req_id}: {str(e)}")
            requirements[req_id]['symbolic'] = None
    
    # Save results
    print(f"Saving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(requirements, f, indent=2)
    
    print("Translation completed successfully!")

if __name__ == "__main__":
    main()