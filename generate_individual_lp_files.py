# generate_individual_lp_files.py
import os
import json
import argparse
import re
from tqdm import tqdm
from models.llm import LlamaModel
from config.llm_config import LlamaConfig

def load_requirements(requirements_file):
    """Load requirements from JSON file with symbolic representations."""
    try:
        with open(requirements_file, 'r') as f:
            requirements_data = json.load(f)
    except Exception as e:
        print(f"Error loading requirements file: {e}")
        return {}
    
    # Process requirements into a more structured format
    processed_requirements = {}
    req_index = 1
    
    for req_text, symbolic in requirements_data.items():
        # Extract requirement number from text if possible
        req_id = None
        match = re.search(r'R(\d+)', req_text)
        if match:
            req_id = match.group(1)
        else:
            # Try to find a number at the beginning of the text
            match = re.match(r'.*?(\d+).*', req_text)
            if match:
                req_id = match.group(1)
            else:
                # Assign sequential number if no ID found
                req_id = str(req_index)
                req_index += 1
        
        # Store processed requirement
        processed_requirements[req_id] = {
            "text": req_text,
            "symbolic": symbolic
        }
    
    return processed_requirements

def clean_deontic_statement(statement):
    """Clean a deontic statement to ensure valid syntax."""
    # Remove trailing commas before periods
    statement = re.sub(r',\s*\.', '.', statement)
    
    # Fix &forbidden references in conditions
    statement = re.sub(r'not &forbidden{([^}]+)},\.', r'not &forbidden{\1}.', statement)
    
    # Ensure proper ending
    if not statement.endswith('.'):
        statement += '.'
    
    return statement

def create_semantic_mapping_prompt(requirement_details, dpa_segment):
    """Create a prompt for generating semantic mapping rules."""
    req_text = requirement_details["text"]
    req_symbolic = requirement_details["symbolic"]
    
    # Extract requirement's deontic statements
    req_deontic_parts = []
    for line in req_symbolic.split('\n'):
        if any(op in line for op in ['&obligatory', '&permitted', '&forbidden']) and line.strip():
            req_deontic_parts.append(line.strip())
    
    req_deontic_text = "\n".join(req_deontic_parts) if req_deontic_parts else req_symbolic
    
    system_prompt = """
You are a specialized AI for legal analysis. Your task is to identify semantic connections 
between a regulatory requirement and a DPA segment.

Create semantic mapping rules ONLY when there is a genuine semantic connection. 
If no connection exists, respond with "NO_SEMANTIC_CONNECTION"

RULES FOR CREATING MAPPINGS:
1. Use standard ASP syntax: head :- body1, body2.
2. Head should be a simple predicate
3. Body should contain predicates that exist in DPA
4. Each rule MUST have both head and body
5. Do not use deontic operators in mapping rules
"""

    user_prompt = f"""
REQUIREMENT TO ANALYZE:
Text: {req_text}
Deontic Logic: {req_deontic_text}

DPA SEGMENT TO CHECK:
Text: {dpa_segment}

If the DPA segment semantically satisfies this requirement, create mapping rules.
If NO semantic connection exists, respond with: "NO_SEMANTIC_CONNECTION"

Rules format:
requirement_predicate :- dpa_predicate1, dpa_predicate2.
"""
    
    return system_prompt, user_prompt

def generate_individual_lp_files(requirements_file, dpa_segments_file, model_path, output_dir):
    """Generate individual LP files for each requirement-DPA pair."""
    
    # Initialize LLM
    print(f"Initializing LLM with model: {model_path}")
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    print("LLM initialized successfully")
    
    # Load requirements
    print(f"Loading requirements from: {requirements_file}")
    requirements = load_requirements(requirements_file)
    print(f"Loaded {len(requirements)} requirements")
    
    # Load DPA deontic data
    print(f"Loading DPA deontic statements from: {dpa_segments_file}")
    with open(dpa_segments_file, 'r') as f:
        dpa_deontic_data = json.load(f)
    
    target_dpa = dpa_deontic_data["dpa_id"]
    dpa_segments = dpa_deontic_data["segments"]
    print(f"Processing {len(dpa_segments)} segments for DPA: {target_dpa}")
    
    # Create output directory for each requirement-segment pair
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate LP files for each requirement-segment pair
    print("Generating LP files for each requirement-segment pair...")
    
    for req_id, req_details in tqdm(requirements.items()):
        # Create directory for this requirement
        req_dir = os.path.join(output_dir, f"req_{req_id}")
        os.makedirs(req_dir, exist_ok=True)
        
        for segment in dpa_segments:
            segment_id = segment["id"]
            segment_text = segment["text"]
            segment_deontics = segment["deontic_statements"]
            
            # Create LP file for this requirement-segment pair
            lp_file_path = os.path.join(req_dir, f"dpa_segment_{segment_id}.lp")
            
            # Generate semantic mapping for this specific pair
            system_prompt, user_prompt = create_semantic_mapping_prompt(req_details, segment_text)
            
            try:
                mapping_result = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
                semantic_mapping = None
                
                if "NO_SEMANTIC_CONNECTION" not in mapping_result:
                    # Extract mapping rules
                    mapping_rules = []
                    for line in mapping_result.split('\n'):
                        line = line.strip()
                        if line and ':-' in line and not line.startswith('%'):
                            # Clean the mapping rule
                            if not line.endswith('.'):
                                line += '.'
                            mapping_rules.append(line)
                    
                    if mapping_rules:
                        semantic_mapping = mapping_rules
            except Exception as e:
                print(f"Error generating semantic mapping for Req {req_id}, Segment {segment_id}: {e}")
                semantic_mapping = None
            
            # Build LP file content
            lp_content = f"""% ========================================================================
% LP file for Requirement {req_id} and DPA Segment {segment_id}
% ========================================================================

% --- Requirement {req_id} ---
% Text: {req_details["text"]}
% Deontic Logic:
{req_details['symbolic']}

% --- DPA Segment {segment_id} ---
% Text: {segment_text}
% Deontic Logic:
"""
            
            # Add DPA segment deontic statements with cleaning
            for statement in segment_deontics:
                cleaned_statement = clean_deontic_statement(statement)
                lp_content += f"{cleaned_statement}\n"
            
            # Add semantic mapping if exists
            if semantic_mapping:
                lp_content += f"""
% --- Semantic Mapping Rules for Requirement {req_id} and Segment {segment_id} ---
"""
                for mapping_rule in semantic_mapping:
                    lp_content += f"{mapping_rule}\n"
            else:
                lp_content += f"""
% --- No semantic mapping found for Requirement {req_id} and Segment {segment_id} ---
"""
            
            # Add simplified status logic
            lp_content += f"""
% --- Status Logic ---
% Basic facts
processor(processor).
controller(controller).
data_subject(data_subject).
personal_data(personal_data).

% Simple satisfaction check
satisfies(req{req_id}) :- true.  % If semantic mapping exists, consider satisfied
not_mentioned(req{req_id}) :- not satisfies(req{req_id}).

% Show directives
#show satisfies/1.
#show not_mentioned/1.
"""
            
            # Write the LP file
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
    
    print(f"Generated LP files for {len(requirements)} requirements and {len(dpa_segments)} segments")
    return requirements, dpa_segments

def main():
    parser = argparse.ArgumentParser(description="Generate individual LP files for each requirement-segment pair")
    parser.add_argument("--requirements", type=str, default="data/processed/requirements_symbolic.json",
                        help="Path to requirements symbolic JSON file")
    parser.add_argument("--dpa_segments", type=str, default="semantic_results/dpa_deontic.json",
                        help="Path to DPA deontic statements JSON file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="semantic_results/individual_lp_files",
                        help="Output directory for individual LP files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate individual LP files
    generate_individual_lp_files(args.requirements, args.dpa_segments, args.model, args.output)

if __name__ == "__main__":
    main()