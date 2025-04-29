# dpa_requirement_comparison.py
import os
import json
import pandas as pd
import re
import random
import tempfile
import subprocess
from datetime import datetime

from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def load_requirements_from_file(file_path):
    """Load requirements from the ground_truth_requirements.txt file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    
    requirements = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove the number prefix (e.g., "1. ", "2. ", etc.)
            cleaned_line = re.sub(r'^\d+\.\s*', '', line)
            if cleaned_line:
                requirements.append(cleaned_line)
    
    return requirements

def run_deolingo(requirement_repr, dpa_repr):
    """
    Run deolingo solver to check if the DPA segment satisfies the requirement.
    
    Args:
        requirement_repr: Symbolic representation of the requirement
        dpa_repr: Symbolic representation of the DPA segment
        
    Returns:
        Result of the deolingo solver execution
    """
    # Create a temporary file for deolingo input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dl', delete=False) as temp:
        # Write the requirement and DPA segment
        temp.write("% Requirement\n")
        temp.write(f"{requirement_repr}\n\n")
        
        temp.write("% DPA Segment\n")
        temp.write(f"{dpa_repr}\n\n")
        
        # Add a query to check if the DPA segment satisfies the requirement
        # Extract the head of the requirement rule (before the ":-" if it exists)
        if ":-" in requirement_repr:
            head = requirement_repr.split(":-")[0].strip()
        else:
            head = requirement_repr.strip()
            
        # Remove the trailing period if it exists
        if head.endswith("."):
            head = head[:-1]
        
        temp.write("% Query to check satisfaction\n")
        temp.write(f"satisfied :- {head}.\n")
        temp.write("?- satisfied.\n")
        
        filename = temp.name
    
    try:
        # Run deolingo on the input file
        result = subprocess.run(
            ["deolingo", filename],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Clean up the temporary file
        os.remove(filename)
        
        # Return the result
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "satisfied": "satisfied" in result.stdout
        }
    
    except Exception as e:
        print(f"Error running deolingo: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return {
            "error": str(e),
            "satisfied": False
        }

def main():
    """
    Select a random requirement and DPA segment, translate them to symbolic
    representations, and run deolingo to check if the segment satisfies the requirement.
    """
    print("Starting DPA-Requirement Comparison")
    
    # Paths to files
    requirements_file = "data/requirements/ground_truth_requirements.txt"
    dpa_file = "data/train_set.csv"
    
    # Check if files exist
    if not os.path.exists(requirements_file):
        print(f"Requirements file not found at {requirements_file}")
        return
    
    if not os.path.exists(dpa_file):
        print(f"DPA file not found at {dpa_file}")
        return
    
    # Get model path from environment or use default
    model_path = os.environ.get("LLAMA_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf")
    
    # Set up LLM configuration
    llm_config = LlamaConfig(
        model_path=model_path,
        temperature=0.1,
    )
    
    # Initialize LLM and translator
    try:
        llm_model = LlamaModel(llm_config)
        print("LLM model initialized successfully")
        translator = DeonticTranslator()
    except Exception as e:
        print(f"Failed to initialize LLM model: {e}")
        return
    
    # Load requirements and select one randomly
    requirements = load_requirements_from_file(requirements_file)
    if not requirements:
        print("No requirements found")
        return
    
    requirement = random.choice(requirements)
    print(f"\n--- Selected Requirement ---")
    print(requirement)
    
    # Load DPA segments and select one randomly
    try:
        df = pd.read_csv(dpa_file)
        if 'Sentence' not in df.columns:
            print(f"'Sentence' column not found in {dpa_file}")
            return
        
        # Select a random DPA segment
        random_row = df.sample(n=1).iloc[0]
        dpa_segment = random_row['Sentence']
        dpa_id = f"dpa_{random_row['ID']}" if 'ID' in df.columns else "dpa_unknown"
        
        print(f"\n--- Selected DPA Segment (ID: {dpa_id}) ---")
        print(dpa_segment)
    except Exception as e:
        print(f"Error loading DPA segments: {e}")
        return
    
    # Translate requirement to symbolic representation
    print("\n--- Translating Requirement ---")
    try:
        req_llm_output = llm_model.generate_symbolic_representation(requirement)
        req_symbolic = translator.translate(req_llm_output)
        print("\nRequirement Symbolic Representation:")
        print(req_symbolic)
    except Exception as e:
        print(f"Error translating requirement: {e}")
        return
    
    # Translate DPA segment to symbolic representation
    print("\n--- Translating DPA Segment ---")
    try:
        dpa_llm_output = llm_model.generate_symbolic_representation(dpa_segment)
        dpa_symbolic = translator.translate(dpa_llm_output)
        print("\nDPA Segment Symbolic Representation:")
        print(dpa_symbolic)
    except Exception as e:
        print(f"Error translating DPA segment: {e}")
        return
    
    # Run deolingo to check if the DPA segment satisfies the requirement
    print("\n--- Running Deolingo Solver ---")
    result = run_deolingo(req_symbolic, dpa_symbolic)
    
    # Print the result
    print("\n--- Deolingo Result ---")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Return Code: {result['returncode']}")
        if result['stdout']:
            print(f"Output:\n{result['stdout']}")
        if result['stderr']:
            print(f"Errors:\n{result['stderr']}")
        
        if result['satisfied']:
            print("\nRESULT: The DPA segment SATISFIES the requirement")
        else:
            print("\nRESULT: The DPA segment does NOT satisfy the requirement")
    
    # Save the results to a JSON file
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"comparison_result_{timestamp}.json")
    
    # Prepare the output
    output_data = {
        "requirement": {
            "text": requirement,
            "symbolic_representation": req_symbolic
        },
        "dpa_segment": {
            "id": dpa_id,
            "text": dpa_segment,
            "symbolic_representation": dpa_symbolic
        },
        "deolingo_result": result,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()