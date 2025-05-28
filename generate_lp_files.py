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

def main():
    parser = argparse.ArgumentParser(description="Generate LP files for DPA segments")
    parser.add_argument("--requirements", type=str, default="results/requirements_deontic_ai_generated.json",
                        help="Path to requirements deontic JSON file")
    parser.add_argument("--dpa", type=str, default="data/test_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Path to LLM model file")
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
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize LLM
    print(f"Initializing LLM with model: {args.model}")
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
            facts = extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, llm_model)
            
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

def extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, llm_model):
    """
    Use LLM to extract facts from a DPA segment based on requirement symbolic representation,
    considering semantic similarity between requirement context and DPA context.
    
    Args:
        segment_text: The text of the DPA segment
        req_text: The text of the requirement
        req_symbolic: The symbolic representation of the requirement
        req_predicates: List of predicates from the requirement
        llm_model: The LLM model to use for extraction
        
    Returns:
        Dictionary mapping predicates to their truth values
    """
    system_prompt = """
     You are a legal-text extractor that converts Data-Processing-Agreement (DPA) segment into Answer-Set-Programming (ASP) facts based on semantic and contextual similarity with GDPR requirement.

Input always contains:

1. "REQUIREMENT" – text of GDPR requirement
2. "SYMBOLIC" - symbolic representation of GDPR requirement in Deontic Logic via ASP
3. "PREDICATES" - the symbolic_atom(s) repeated from a symbolic representation of the requirement, semicolon-separated.
4. "CLAUSE" - a single DPA segment.

Analyze the SYMBOLIC representation of regulatory requirements to understand the structure of the requirement. Understand which textual parts of requirement relate to each predicates/atoms from a symbolic form. Understand the semantic meaning of each predicate in the context of regulatory requirement.

 Emit an ACTION predicate only if the clause contains BOTH:
 1. a verb that matches the action (e.g. “make available”, “provide”, “supply” for provide_compliance_information)
 2. an object / purpose that matches the predicate’s intended goal
    – for provide_compliance_information this means wording about “demonstrate / evidence / prove compliance”, “Article 28”, “records of processing”, “documentation of GDPR compliance”.

TASK:
- Decide which (if any) of the listed predicates the clause semantically entails.
- Output all entailed by CLAUSE predicates separated by ;
- If none are entailed, output exactly NO_FACTS.
- Produce nothing else: no prose, no JSON, no comments.
- If in the CLAUSE text a predicate from PREDICATES is explicitly violated in the text return it with - sign before it (e.g. -encrypt_data)

Examples:
Example 1:
REQUIREMENT: Processor must ensure that all authorised personnel are bound by confidentiality obligations.
SYMBOLIC: &obligatory{ensure_confidentiality} :- role(processor).
PREDICATES: ensure_confidentiality; role(processor)
CLAUSE: The Processor shall ensure that every employee authorised to process Customer Personal Data is subject to a contractual duty of confidentiality.
Expected output: ensure_confidentiality; role(processor)

Example 2:
REQUIREMENT: Processor must encrypt personal data during transmission and at rest.
SYMBOLIC: &obligatory{encrypt_data} :- role(processor).
PREDICATES: encrypt_data; role(processor)
CLAUSE: This DPA shall remain in effect so long as processor Processes Personal Data, notwithstanding the expiration or termination of the Agreement.
Expected output: role(processor)

Example 3:
REQUIREMENT: The processor must encrypt all the data collected from customers.
SYMBOLIC: &obligatory{encrypt_collected_data} :- role(processor)
PREDICATES: encrypt_collected_data; role(processor)
CLAUSE: The processor will store customer's data in raw format.
Expected output: -encrypt_collected_data; role(processor)

Example 4:
REQUIREMENT: The processor must notify controller about data breaches.
SYMBOLIC: &obligatory{notyfy_controller_data_breaches} :- role(processor)
PREDICATES: notyfy_controller_data_breaches; role(processor)
CLAUSE: Sub-Processor rights
Expected output: NO_FACTS
    """
    
    user_prompt = f""" REQUIREMENT: {req_text} SYMBOLIC: {req_symbolic} PREDICATES: {'; '.join(req_predicates)} CLAUSE: {segment_text}
"""
    
    # Get facts from LLM
    response = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
    
    # Process the response
    facts = {}
    
    # Split response by semicolon and process each fact
    for fact in response.strip().split(';'):
        fact = fact.strip()
        if not fact:
            continue
            
        # Remove any trailing period
        if fact.endswith('.'):
            fact = fact[:-1]
            
        # Skip if it's NO_FACTS
        if fact == "NO_FACTS":
            continue
            
        # Handle negative predicates
        if fact.startswith('-'):
            # Remove any space after the minus sign
            if len(fact) > 1 and fact[1] == ' ':
                fact = '-' + fact[1:].strip()            
        # Add the fact as True
        facts[fact] = True
    
    return facts

def generate_lp_file(req_symbolic, facts, req_predicates, req_text, segment_text):
    """Generate a complete LP file with the correct template structure."""
    # Clean up the symbolic representation to ensure valid syntax
    clean_symbolic = req_symbolic
    # Replace not(predicate) with -predicate for better compatibility
    clean_symbolic = re.sub(r'not\s*\(([^)]+)\)', r'-\1', clean_symbolic)
    
    # Extract predicates from rule body
    body_predicates = set()
    if ':-' in clean_symbolic:
        body = clean_symbolic.split(':-')[1].strip()
        if body.endswith('.'):
            body = body[:-1]
        for pred in body.split(','):
            pred = pred.strip()
            # Remove 'not' prefix if present
            if pred.startswith('not '):
                pred = pred[4:].strip()
            # Remove '-' prefix if present
            elif pred.startswith('-'):
                pred = pred[1:].strip()
            if pred:
                body_predicates.add(pred)
    
    # Add requirement and DPA segment text as comments
    lp_content = f"""% Requirement Text:
% {req_text}
%
% DPA Segment:
% {segment_text}
%
% 0. External declarations for rule body predicates
"""
    
    # Add external declarations for body predicates
    for pred in sorted(body_predicates):
        lp_content += f"#external {pred}.\n"
    
    lp_content += """
% 1. Normative layer
"""
    
    # Add the symbolic rule
    lp_content += f"{clean_symbolic}\n\n"
    
    # Add facts from the DPA segment
    lp_content += """% 2. Facts extracted from DPA segment
"""
    
    # Add facts from the DPA segment, ensuring no duplicates and valid syntax
    added_facts = False
    seen_facts = set()
    for predicate, status in facts.items():
        if predicate != "NO_FACTS":  # Explicitly exclude NO_FACTS
            # Clean up predicate to avoid syntax errors
            clean_predicate = predicate
            
            # Replace not() with minus sign but preserve balanced parentheses
            if 'not(' in clean_predicate:
                clean_predicate = re.sub(r'not\s*\(([^)]+)\)', r'-\1', clean_predicate)
            
            # Fix unclosed parentheses instead of removing all closing parentheses
            open_count = clean_predicate.count('(')
            close_count = clean_predicate.count(')')
            
            if open_count > close_count:
                # Add missing closing parentheses
                clean_predicate += ')' * (open_count - close_count)
            elif open_count < close_count and not clean_predicate.startswith('-'):
                # Too many closing parentheses, remove extras
                excess = close_count - open_count
                for _ in range(excess):
                    clean_predicate = clean_predicate.rstrip(')')
            
            # Remove trailing period if it exists
            if clean_predicate.endswith('.'):
                clean_predicate = clean_predicate[:-1]
            
            if status is True and clean_predicate not in seen_facts:  # Positively mentioned
                lp_content += f"{clean_predicate}.\n"
                seen_facts.add(clean_predicate)
                added_facts = True
            elif status is False and clean_predicate not in seen_facts:  # Negatively mentioned
                # Ensure double negation doesn't occur
                if clean_predicate.startswith('-'):
                    # Remove the minus sign to avoid double negation
                    lp_content += f"{clean_predicate[1:]}.\n"
                else:
                    lp_content += f"-{clean_predicate}.\n"
                seen_facts.add(clean_predicate)
                added_facts = True
    
    # If no facts were added, add a comment
    if not added_facts:
        lp_content += "% No semantically relevant facts found in this segment\n"
    
    lp_content += "\n"
    
    # Add status mapping rules
    lp_content += """% 3. Map Deolingo's internal status atoms to our labels
"""
    
    # Extract obligation names from the symbolic representation
    obligation_pattern = r'&obligatory{([^}]+)}'
    obligations = re.findall(obligation_pattern, clean_symbolic)
    forbidden_pattern = r'&forbidden{([^}]+)}'
    forbiddens = re.findall(forbidden_pattern, clean_symbolic)
    permitted_pattern = r'&permitted{([^}]+)}'
    permitted = re.findall(permitted_pattern, clean_symbolic)
    
    deontic_entities = obligations + forbiddens + permitted
    
    if not deontic_entities:
        # If no obligations found, add a default status mapping
        lp_content += """status(satisfied)     :- &fulfilled_obligation{default}.
status(violated)      :- &violated_obligation{default}.
status(not_mentioned) :- &undetermined_obligation{default}.
"""
    else:
        # Add mapping rules for each deontic entity
        for entity in deontic_entities:
            lp_content += f"status(satisfied)     :- &fulfilled_obligation{{{entity}}}.\n"
            lp_content += f"status(violated)      :- &violated_obligation{{{entity}}}.\n"
            lp_content += f"status(not_mentioned) :- &undetermined_obligation{{{entity}}}.\n\n"
    
    # Add show directive
    lp_content += """#show status/1.
"""
    
    return lp_content

if __name__ == "__main__":
    main()