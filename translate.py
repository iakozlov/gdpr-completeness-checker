# translate.py
import os
import re
import pandas as pd
from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def init_llm(model_path):
    """Initialize LLM and translator."""
    llm_config = LlamaConfig(model_path=model_path)
    llm_model = LlamaModel(llm_config)
    translator = DeonticTranslator()
    return llm_model, translator

def translate_requirements(file_path, model_path):
    """Translate all requirements from the file to symbolic representation."""
    # Initialize LLM
    llm_model, translator = init_llm(model_path)
    
    # Load requirements
    requirements = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract requirement ID and text
            parts = line.split('. ', 1)
            if len(parts) == 2 and parts[0].isdigit():
                req_id = parts[0]
                req_text = parts[1]
                
                # Translate to symbolic representation
                print(f"Translating requirement {req_id}...")
                llm_output = llm_model.generate_symbolic_representation(req_text)
                symbolic = translator.translate(llm_output)
                
                # Store in results
                requirements[req_id] = {
                    "text": req_text,
                    "symbolic": symbolic
                }
    
    return requirements

def translate_dpa_segments(dpa_groups, model_path):
    """Translate DPA segments grouped by DPA to symbolic representation."""
    # Initialize LLM
    llm_model, translator = init_llm(model_path)
    
    # Process each DPA group
    dpa_translations = {}
    
    for dpa_id, group in dpa_groups:
        print(f"Translating DPA {dpa_id}...")
        
        # Extract segments
        segments = []
        for _, row in group.iterrows():
            segment_id = row["ID"]
            segment_text = row["Sentence"]
            
            # Translate to symbolic representation
            llm_output = llm_model.generate_symbolic_representation(segment_text)
            symbolic = translator.translate(llm_output)
            
            # Store segment with translation
            segments.append({
                "segment_id": segment_id,
                "text": segment_text,
                "symbolic": symbolic,
                "requirements": [
                    row.get("Requirement-1", ""),
                    row.get("Requirement-2", ""),
                    row.get("Requirement-3", "")
                ]
            })
        
        # Store in results
        dpa_translations[dpa_id] = segments
    
    return dpa_translations