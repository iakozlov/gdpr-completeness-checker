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

def filter_think_sections(text):
    """
    Remove <think> sections from model responses.
    
    Args:
        text (str): The raw model response
        
    Returns:
        str: The filtered text with <think> sections removed
    """
    # Use regex to remove everything between <think> and </think> tags (case insensitive, multiline)
    filtered = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra whitespace that might be left
    filtered = re.sub(r'\n\s*\n', '\n', filtered.strip())
    
    return filtered

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
    parser.add_argument("--target_dpa", type=str, default="Online 126",
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
    
    # Process each requirement
    for req_id, req_info in tqdm(requirements.items(), desc="Processing requirements"):
        # Skip if debug_req_id is specified and this isn't it
        if args.debug_req_id and req_id != args.debug_req_id:
            continue
            
        req_text = req_info["text"]
        req_symbolic = req_info["symbolic"]
        
        # Create directory for this requirement
        req_dir = os.path.join(dpa_dir, f"req_{req_id}")
        os.makedirs(req_dir, exist_ok=True)
        
        # Extract predicates from the requirement
        req_predicates = extract_predicates(req_info)
        
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
            
            # Generate LP file content
            lp_content = generate_lp_file(req_symbolic, facts, req_predicates, req_text, segment_text)
            
            if args.verbose:
                print(f"Generated LP content:\n{lp_content}")
            
            # Write to file
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
    
    print("LP file generation complete!")

def extract_predicates(req_info):
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
    system_prompt = """You are a meticulous and cautious AI legal analyst specializing in GDPR compliance. Your task is to verify if a DPA clause explicitly and fully satisfies the conditions and obligations described in a GDPR requirement.

INPUT FORMAT:
1.  **REQUIREMENT:** The full text of the GDPR requirement.
2.  **SYMBOLIC:** The requirement's formal representation in Answer Set Programming (ASP).
3.  **PREDICATES:** The list of relevant atomic facts (atoms) from the symbolic rule.
4.  **CLAUSE:** A single text segment from the DPA.

TASK:
Based on a rigorous analysis, identify which predicates from the provided list are explicitly and unambiguously supported by the text in the CLAUSE. Output only the supported predicate names, separated by a semicolon.

CRITICAL REASONING FRAMEWORK (Follow these steps):

1.  **Direct Evidence Rule:** A predicate can only be confirmed if there is direct, explicit evidence in the CLAUSE. Do not make assumptions or infer meaning beyond what is written. The clause must contain the full semantic weight of the predicate.

2.  **Scope & Conditionality Analysis:**
    - Does the CLAUSE make a firm, unconditional commitment (e.g., 'The processor shall...')?
    - Or is the commitment conditional or limited (e.g., '...if requested by the controller', '...where possible', '...may assist')?
    - If the REQUIREMENT implies an unconditional obligation but the CLAUSE is conditional or limited, the head predicate of the rule is NOT satisfied.

3.  **Action vs. Mechanism Distinction:**
    - Does the CLAUSE state a commitment to *perform an action* (e.g., 'The processor will notify...')?
    - Or does it merely describe the *existence of a process or mechanism* (e.g., 'The processor has a process for notification...')?
    - Describing a mechanism does not satisfy an obligation to perform the action itself. Only confirm the predicate if the action is committed to.

4.  **Functional Equivalence over Keywords:** Do not rely on simple keyword matching. Focus on the legal and functional outcome. For example, a clause stating that changes are 'added to a public list available to the controller' can be functionally equivalent to 'informing' the controller.

OUTPUT INSTRUCTIONS:
-   Output the names of the confirmed predicates, separated by a semicolon (e.g., `role(processor); has_general_written_authorization`).
-   If the CLAUSE explicitly describes an action that violates a predicate, prefix the predicate with a hyphen (e.g., `-ensures_personnel_are_bound_by_confidentiality`).
-   **Golden Rule: If, after following all reasoning steps, you have any doubt, or if no predicates are explicitly and fully supported, you MUST output the exact string `NO_FACTS`. It is better to miss a fact than to wrongly confirm one.**
-   Do NOT add any explanations, notes, or apologies.

--- EXAMPLES ---
Example 1:
REQUIREMENT: The processor shall ensure that persons authorised to process personal data have committed themselves to confidentiality or are under an appropriate statutory obligation of confidentiality.
SYMBOLIC: &obligatory{ensures_personnel_are_bound_by_confidentiality} :- role(processor).
PREDICATES: ensures_personnel_are_bound_by_confidentiality; role(processor)
CLAUSE: The Processor shall ensure that every employee authorized to process Customer Personal Data is subject to a contractual duty of confidentiality.
Expected output: ensures_personnel_are_bound_by_confidentiality; role(processor)

Example 2:
REQUIREMENT: The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller.
SYMBOLIC: &forbidden{engages_subprocessor} :- role(processor), not has_prior_specific_authorization, not has_prior_general_authorization.
PREDICATES: engages_subprocessor; role(processor); has_prior_specific_authorization; has_prior_general_authorization
CLAUSE: Where processor authorises any sub-processor as described in Section 6.1
Expected output: role(processor)

Example 3:
REQUIREMENT: The processor must encrypt all the data collected from customers.
SYMBOLIC: &obligatory{implements_encryption} :- role(processor).
PREDICATES: implements_encryption; role(processor)
CLAUSE: The processor will store customer's data in raw format.
Expected output: -implements_encryption; role(processor)

Example 4:
REQUIREMENT: The processor must notify controller about data breaches.
SYMBOLIC: &obligatory{notifies_controller_of_data_breach} :- role(processor).
PREDICATES: notifies_controller_of_data_breach; role(processor)
CLAUSE: Sub-Processor rights
Expected output: NO_FACTS"""

    
    user_prompt = f""" REQUIREMENT: {req_text} SYMBOLIC: {req_symbolic} PREDICATES: {'; '.join(req_predicates)} CLAUSE: {segment_text}"""
    
    if use_ollama:
        response = llm_model.generate(user_prompt, model_name=model_name, system_prompt=system_prompt)
    else:
        # For non-Ollama models, combine system and user prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = llm_model.generate(combined_prompt)
    
    # Filter out <think> sections from reasoning models like qwen3
    response = filter_think_sections(response)
    
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
    lp_content = f"% Requirement Text:\n% {req_text}\n%\n"
    lp_content += f"% DPA Segment:\n% {segment_text}\n%\n"
    
    # Extract body atoms from the symbolic rule
    body_atoms = extract_body_atoms(req_symbolic)
    
    # Add external declarations only for body atoms
    if body_atoms:
        lp_content += "% External declarations for rule body predicates\n"
        for atom in body_atoms:
            lp_content += f"#external {atom}.\n"
        lp_content += "\n"
    
    # Add the requirement's symbolic representation (normative layer)
    lp_content += "% 1. Normative layer\n"
    lp_content += f"{req_symbolic}\n\n"
    
    # Add facts
    lp_content += "% 2. Facts extracted from DPA segment\n"
    if facts:
        for pred, value in facts.items():
            if value:
                lp_content += f"{pred}.\n"
            else:
                lp_content += f"-{pred}.\n"
    else:
        lp_content += "% No semantically relevant facts found in this segment\n"
    
    lp_content += "\n"
    
    # Add status mapping - determine the deontic operator from the symbolic rule
    lp_content += "% 3. Map Deolingo's internal status atoms to our labels\n"
    
    # Extract the deontic operator and predicate from the symbolic rule
    if "&obligatory{" in req_symbolic:
        # Extract predicate from &obligatory{predicate}
        predicate = req_symbolic.split("&obligatory{")[1].split("}")[0]
        lp_content += f"status(satisfied)     :- &fulfilled_obligation{{{predicate}}}.\n"
        lp_content += f"status(violated)      :- &violated_obligation{{{predicate}}}.\n"
        lp_content += f"status(not_mentioned) :- &undetermined_obligation{{{predicate}}}.\n"
    elif "&forbidden{" in req_symbolic:
        # Extract predicate from &forbidden{predicate}
        predicate = req_symbolic.split("&forbidden{")[1].split("}")[0]
        lp_content += f"status(satisfied)     :- &fulfilled_prohibition{{{predicate}}}.\n"
        lp_content += f"status(violated)      :- &violated_prohibition{{{predicate}}}.\n"
        lp_content += f"status(not_mentioned) :- &undetermined_prohibition{{{predicate}}}.\n"
    elif "&permitted{" in req_symbolic:
        # Extract predicate from &permitted{predicate}
        predicate = req_symbolic.split("&permitted{")[1].split("}")[0]
        lp_content += f"status(satisfied)     :- &fulfilled_permission{{{predicate}}}.\n"
        lp_content += f"status(violated)      :- &violated_permission{{{predicate}}}.\n"
        lp_content += f"status(not_mentioned) :- &undetermined_permission{{{predicate}}}.\n"
    else:
        # Fallback for unknown deontic operators
        lp_content += "% Warning: Unknown deontic operator in symbolic rule\n"
        lp_content += "status(not_mentioned) :- true.\n"
    
    lp_content += "\n#show status/1.\n"
    
    return lp_content

if __name__ == "__main__":
    main()
