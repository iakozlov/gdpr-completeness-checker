# generate_lp_files.py
import os
import json
import argparse
import pandas as pd
import re
from tqdm import tqdm
from models.gpt_model import GPTModel
from config.gpt_config import GPTConfig
from models.llama_model import LlamaModel
from config.llama_config import LlamaConfig
from ollama_client import OllamaClient

def main():
    parser = argparse.ArgumentParser(description="Generate LP files for DPA segments")
    parser.add_argument("--requirements", type=str, default="results/requirements_deontic_ai_generated.json",
                        help="Path to requirements deontic JSON file")
    parser.add_argument("--dpa", type=str, default="data/test_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use (gpt-4o-mini, llama2-70b, mistral-7b, etc.)")
    parser.add_argument("--output", type=str, default="results/lp_files",
                        help="Output directory for LP files")
    parser.add_argument("--target_dpa", type=str, default="Online 124",
                        help="Target DPA to process (default: Online 124)")
    parser.add_argument("--req_ids", type=str, default="all",
                        help="Comma-separated list of requirement IDs to process, or 'all' (default: all)")
    parser.add_argument("--max_segments", type=int, default=0,
                        help="Maximum number of segments to process (0 means all, default: 0)")
    parser.add_argument("--debug_req_id", type=str,
                        help="Debug mode: Process only this specific requirement ID")
    parser.add_argument("--debug_segment_id", type=str, 
                        help="Debug mode: Process only this specific segment ID")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for debugging")
    parser.add_argument("--use_ollama", action="store_true",
                        help="Use Ollama for model inference")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize LLM
    print(f"Initializing LLM with model: {args.model}")
    if args.use_ollama:
        llm_model = OllamaClient()
        if not llm_model.check_health():
            print("Error: Ollama server is not running. Please start it first.")
            return
    else:
        llm_config = LlamaConfig(model=args.model, temperature=0.1)
        llm_model = LlamaModel(llm_config)
    print("LLM initialized successfully")
    
    # Load requirements
    print(f"Loading requirements from: {args.requirements}")
    with open(args.requirements, 'r') as f:
        all_requirements = json.load(f)
    
    # Filter requirements by ID if specified
    if args.req_ids.lower() != "all":
        req_ids = [id.strip() for id in args.req_ids.split(",")]
        requirements = {id: all_requirements[id] for id in req_ids if id in all_requirements}
        if not requirements:
            print(f"Error: No valid requirement IDs found. Available IDs: {', '.join(all_requirements.keys())}")
            return
        print(f"Processing {len(requirements)} requirements with IDs: {', '.join(requirements.keys())}")
    else:
        requirements = all_requirements
        print(f"Processing all {len(requirements)} requirements")
    
    # Load DPA segments
    print(f"Loading DPA segments from: {args.dpa}")
    df = pd.read_csv(args.dpa)
    
    # Filter for the target DPA only
    target_dpa = args.target_dpa
    df_filtered = df[df['DPA'] == target_dpa]
    
    if df_filtered.empty:
        print(f"Error: DPA '{target_dpa}' not found in the dataset.")
        return
    
    # Apply segment limit if specified
    if args.max_segments > 0:
        df_filtered = df_filtered.head(args.max_segments)
        print(f"Processing first {len(df_filtered)} segments for DPA: {target_dpa}")
    else:
        print(f"Processing all {len(df_filtered)} segments for DPA: {target_dpa}")
    
    # Create directory for this DPA
    dpa_dir = os.path.join(args.output, f"dpa_{target_dpa.replace(' ', '_')}")
    os.makedirs(dpa_dir, exist_ok=True)
    
    # Process each requirement
    for req_id, req_info in tqdm(requirements.items(), desc="Processing requirements"):
        # Skip if debug_req_id is specified and this isn't it
        if args.debug_req_id and req_id != args.debug_req_id:
            continue
            
        req_text = req_info["text"]
        req_symbolic = req_info["symbolic"]
        
        if args.verbose:
            print(f"\nProcessing requirement {req_id}:")
            print(f"Text: {req_text}")
            print(f"Symbolic: {req_symbolic}")
        
        # Create directory for this requirement
        req_dir = os.path.join(dpa_dir, f"req_{req_id}")
        os.makedirs(req_dir, exist_ok=True)
        
        # Extract predicates from the requirement
        req_predicates = extract_predicates(req_symbolic, req_info)
        if args.verbose:
            print(f"Extracted predicates: {req_predicates}")
        
        # Process each segment
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing segments for requirement {req_id}"):
            segment_id = row["ID"]
            
            # Skip if debug_segment_id is specified and this isn't it
            if args.debug_segment_id and str(segment_id) != args.debug_segment_id:
                continue
                
            segment_text = row["Sentence"]
            
            if args.verbose:
                print(f"\nProcessing segment {segment_id}:")
                print(f"Text: {segment_text}")
            
            # Generate LP file for this segment
            lp_file_path = os.path.join(req_dir, f"segment_{segment_id}.lp")
            
            # Extract facts from DPA segment
            facts = extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, llm_model, args.use_ollama, args.model)
            
            if args.verbose:
                print(f"Extracted facts: {facts}")
            
            # Generate LP file content
            lp_content = generate_lp_file(req_symbolic, facts, req_predicates, req_text, segment_text)
            
            if args.verbose:
                print(f"Generated LP content:\n{lp_content}")
            
            # Write to file
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
            
            if args.verbose:
                print(f"Written to: {lp_file_path}")
    
    print("LP file generation complete!")

def extract_predicates(symbolic, req_info):
    """Extract predicates from the requirement's atoms field."""
    # Return the atoms directly from the requirement info
    return req_info.get("atoms", [])

def extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, llm_model, use_ollama=False, model_name="llama3.3:70b"):
    """
    Use LLM to extract facts from a DPA segment based on requirement symbolic representation,
    considering semantic similarity between requirement context and DPA context.
    
    Args:
        segment_text: The text of the DPA segment
        req_text: The text of the requirement
        req_symbolic: The symbolic representation of the requirement
        req_predicates: List of predicates from the requirement
        llm_model: The LLM model to use for extraction
        use_ollama: Whether to use Ollama for inference
        model_name: Name of the model to use
        
    Returns:
        Dictionary mapping predicates to their truth values
    """
    system_prompt = """You are a legal-text extractor that converts Data-Processing-Agreement (DPA) segment into Answer-Set-Programming (ASP) facts based on semantic and contextual similarity with GDPR requirement.

Analyze the SYMBOLIC representation of regulatory requirements to understand the structure of the requirement. Understand which textual parts of requirement relate to each predicates/atoms from a symbolic form. Understand the semantic meaning of each predicate in the context of regulatory requirement.

Emit an ACTION predicate only if the clause contains BOTH:
1. a verb that matches the action (e.g. "make available", "provide", "supply" for provide_compliance_information)
2. an object / purpose that matches the predicate's intended goal
   – for provide_compliance_information this means wording about "demonstrate / evidence / prove compliance", "Article 28", "records of processing", "documentation of GDPR compliance".

TASK:
- Decide which (if any) of the listed predicates the clause semantically entails.
- Output all entailed by CLAUSE predicates separated by ;
- If none are entailed, output exactly NO_FACTS.
- Produce nothing else: no prose, no JSON, no comments.
- If in the CLAUSE text a predicate from PREDICATES is explicitly violated in the text return it with - sign before it (e.g. -encrypt_data)"""
    
    user_prompt = f"""REQUIREMENT: {req_text}
SYMBOLIC: {req_symbolic}
PREDICATES: {';'.join(req_predicates)}
CLAUSE: {segment_text}"""
    
    if use_ollama:
        response = llm_model.generate(user_prompt, model_name=model_name, system_prompt=system_prompt)
    else:
        # For non-Ollama models, combine system and user prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = llm_model.generate(combined_prompt)
    
    # Parse the response
    if response.strip() == "NO_FACTS":
        return {}
    
    facts = {}
    for pred in response.strip().split(';'):
        pred = pred.strip()
        if pred.startswith('-'):
            facts[pred[1:]] = False
        else:
            facts[pred] = True
    
    return facts

def extract_body_atoms(symbolic_rule):
    """Extract atoms from the body of an ASP rule.
    
    Args:
        symbolic_rule (str): ASP rule in the format "head :- body."
        
    Returns:
        List[str]: List of atoms in the body
    """
    if ":-" not in symbolic_rule:
        return []
    
    # Split on :- to get the body part
    parts = symbolic_rule.split(":-")
    if len(parts) < 2:
        return []
    
    body = parts[1].strip()
    
    # Remove the trailing period
    if body.endswith('.'):
        body = body[:-1]
    
    # Split by comma to get individual atoms
    atoms = []
    for atom in body.split(','):
        atom = atom.strip()
        
        # Remove 'not ' prefix if present
        if atom.startswith('not '):
            atom = atom[4:].strip()
        
        # Add atom if it's not empty
        if atom:
            atoms.append(atom)
    
    return atoms

def generate_lp_file(req_symbolic, facts, req_predicates, req_text, segment_text):
    """Generate the content of an LP file."""
    # Start with the requirement's symbolic representation
    lp_content = f"% Requirement: {req_text}\n"
    lp_content += f"% Symbolic: {req_symbolic}\n"
    lp_content += f"% Segment: {segment_text}\n\n"
    
    # Extract body atoms from the symbolic rule
    body_atoms = extract_body_atoms(req_symbolic)
    
    # Add external declarations only for body atoms
    if body_atoms:
        lp_content += "% External declarations for body atoms:\n"
        for atom in body_atoms:
            lp_content += f"#external {atom}.\n"
        lp_content += "\n"
    
    # Add the requirement's symbolic representation
    lp_content += f"{req_symbolic}\n\n"
    
    # Add facts
    if facts:
        lp_content += "% Facts extracted from the segment:\n"
        for pred, value in facts.items():
            if value:
                lp_content += f"{pred}.\n"
            else:
                lp_content += f"not {pred}.\n"
    else:
        lp_content += "% No facts extracted from the segment.\n"
    
    return lp_content

if __name__ == "__main__":
    main()