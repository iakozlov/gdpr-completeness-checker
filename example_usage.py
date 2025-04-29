# example_usage.py
import os
import json
import re
from datetime import datetime

# Import from our main project structure
from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def load_requirements_from_file(file_path):
    """
    Load requirements from the ground_truth_requirements.txt file.
    Handles the numbered format and returns a list of requirement texts.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    
    requirements = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
                
            # Remove the number prefix (e.g., "1. ", "2. ", etc.)
            # Match patterns like "1. " or "19. "
            cleaned_line = re.sub(r'^\d+\.\s*', '', line)
            
            if cleaned_line:
                requirements.append(cleaned_line)
    
    return requirements

def main():
    """
    Demonstrate translation of GDPR requirements to symbolic representations
    using all requirements from ground_truth_requirements.txt file.
    """
    print("Starting GDPR Requirements Translator")
    
    # Path to the requirements file
    requirements_file = "data/requirements/ground_truth_requirements.txt"
    
    # Check if file exists
    if not os.path.exists(requirements_file):
        print(f"Requirements file not found at {requirements_file}")
        print("Please ensure the file exists or update the path.")
        return
    
    # Get model path from environment or use default
    model_path = os.environ.get("LLAMA_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf")
    
    # Set up LLM configuration
    llm_config = LlamaConfig(
        model_path=model_path,
        temperature=0.1,  # Lower temperature for more deterministic outputs
    )
    
    # Initialize LLM and translator
    try:
        llm_model = LlamaModel(llm_config)
        print("LLM model initialized successfully")
        translator = DeonticTranslator()
    except Exception as e:
        print(f"Failed to initialize LLM model: {e}")
        return
    
    # Load requirements from file
    requirements = load_requirements_from_file(requirements_file)
    
    if not requirements:
        print("No requirements found in the file. Please check the file content.")
        return
    
    # Create dictionary to store results
    translations = {}
    
    # Process each requirement
    print(f"Processing {len(requirements)} requirements from {requirements_file}")
    
    for i, requirement in enumerate(requirements, 1):
        print(f"\n--- Requirement {i}/{len(requirements)} ---")
        print(f"Text: {requirement}")
        
        # Generate symbolic representation using the same method as in main.py
        try:
            # System prompt optimized for Instruct models
            system_prompt = (
                "You are a specialized AI assistant trained to translate GDPR legal requirements "
                "into formal Deontic Logic representations compatible with Answer Set Programming (ASP). "
                "You must follow the exact formalism required, with no additional explanations."
            )
            
            llm_output = llm_model.generate_symbolic_representation(requirement, system_prompt)
            print("\nLLM Output:")
            print("===========")
            print(llm_output)
            
            # Translate to proper format for deolingo
            translated = translator.translate(llm_output)
            print("\nTranslated for deolingo:")
            print("========================")
            print(translated)
            print("\n" + "=" * 70)
            
            # Store in translations dictionary
            translations[requirement] = translated
            
        except Exception as e:
            print(f"Failed to process requirement {i}: {e}")
    
    # Save translations to JSON file
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"requirement_translations_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(translations, f, indent=2)
    
    print(f"\nRequirements translation complete. Results saved to {output_file}")
    print(f"Processed {len(translations)} requirements successfully")

if __name__ == "__main__":
    main()