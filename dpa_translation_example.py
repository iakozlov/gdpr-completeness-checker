# dpa_translation_example.py
import os
import json
import pandas as pd
from datetime import datetime

from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def main():
    """
    Example script to translate a subset of DPA segments from train_set.csv
    to symbolic representations and save the results to a JSON file.
    """
    print("Starting DPA Segment Translator")
    
    # Path to the DPA segments CSV file
    csv_file = "data/train_set.csv"
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"CSV file not found at {csv_file}")
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
    
    # Load DPA segments from CSV (subset of 10 random segments)
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Verify 'Sentence' column exists
        if 'Sentence' not in df.columns:
            print(f"Error: 'Sentence' column not found in {csv_file}")
            return
        
        # Select a random subset (10 segments)
        sample_size = min(10, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        print(f"Selected {sample_size} random DPA segments from {csv_file}")
        
        # Create dictionary to store results
        translations = {}
        
        # Process each segment
        for i, row in df_sample.iterrows():
            segment_text = row['Sentence']
            segment_id = f"dpa_{row['ID']}" if 'ID' in df.columns else f"dpa_{i}"
            
            print(f"\n--- DPA Segment {i+1}/{sample_size} ---")
            print(f"ID: {segment_id}")
            print(f"Text: {segment_text}")
            
            # Generate symbolic representation
            try:
                # System prompt optimized for Instruct models
                system_prompt = (
                    "You are a specialized AI assistant trained to translate DPA segments "
                    "into formal Deontic Logic representations compatible with Answer Set Programming (ASP). "
                    "You must follow the exact formalism required, with no additional explanations."
                )
                
                llm_output = llm_model.generate_symbolic_representation(segment_text, system_prompt)
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
                translations[segment_id] = {
                    "original_text": segment_text,
                    "symbolic_representation": translated
                }
                
            except Exception as e:
                print(f"Failed to process segment {segment_id}: {e}")
        
        # Save translations to JSON file
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"dpa_translations_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(translations, f, indent=2)
        
        print(f"\nDPA translations complete. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing DPA segments: {e}")

if __name__ == "__main__":
    main()