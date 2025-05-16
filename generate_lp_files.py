# generate_lp_files.py
import os
import json
import argparse
import pandas as pd
import re
from tqdm import tqdm
from models.llm import LlamaModel
from config.llm_config import LlamaConfig

def main():
    parser = argparse.ArgumentParser(description="Generate LP files for DPA segments")
    parser.add_argument("--requirements", type=str, default="results/requirements_deontic.json",
                        help="Path to requirements deontic JSON file")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="results/lp_files",
                        help="Output directory for LP files")
    parser.add_argument("--target_dpa", type=str, default="Online 1",
                        help="Target DPA to process (default: Online 1)")
    parser.add_argument("--req_ids", type=str, default="all",
                        help="Comma-separated list of requirement IDs to process, or 'all' (default: all)")
    parser.add_argument("--max_segments", type=int, default=0,
                        help="Maximum number of segments to process (0 means all, default: 0)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize LLM
    print(f"Initializing LLM with model: {args.model}")
    llm_config = LlamaConfig(model_path=args.model, temperature=0.1)
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
        req_text = req_info["text"]
        req_symbolic = req_info["symbolic"]
        
        # Create directory for this requirement
        req_dir = os.path.join(dpa_dir, f"req_{req_id}")
        os.makedirs(req_dir, exist_ok=True)
        
        # Extract predicates from the requirement
        req_predicates = extract_predicates(req_symbolic)
        
        # Process each segment
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing segments for requirement {req_id}"):
            segment_id = row["ID"]
            segment_text = row["Sentence"]
            
            # Generate LP file for this segment
            lp_file_path = os.path.join(req_dir, f"segment_{segment_id}.lp")
            
            # Extract facts from DPA segment
            facts = extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, llm_model)
            
            # Generate LP file content
            lp_content = generate_lp_file(req_symbolic, facts, req_predicates, req_text, segment_text)
            
            # Write to file
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
    
    print("LP file generation complete!")

def extract_predicates(symbolic):
    """Extract all predicates from a symbolic representation."""
    predicates = set()
    
    # Extract predicates from deontic operators
    for op in ['&obligatory', '&permitted', '&forbidden']:
        pattern = rf'{op}{{([^}}]+)}}'
        matches = re.findall(pattern, symbolic)
        for match in matches:
            # Extract predicate without arguments
            if '(' in match:
                predicate = match.split('(')[0].strip()
                predicates.add(predicate)
            else:
                predicates.add(match.strip())
    
    # Extract predicates from conditions
    for line in symbolic.split('\n'):
        if ':-' in line:
            condition = line.split(':-')[1].strip()
            if condition.endswith('.'):
                condition = condition[:-1]
            
            # Split by comma and extract predicates
            for part in condition.split(','):
                part = part.strip()
                if part.startswith('not '):
                    part = part[4:].strip()
                
                # Extract predicate without arguments
                if '(' in part:
                    predicate = part.split('(')[0].strip()
                    predicates.add(predicate)
                else:
                    predicates.add(part)
    
    return list(predicates)

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
    You are a legal-text extractor that converts Data-Processing-Agreement (DPA) segment into ASP facts based on semantic and contextual similarity with GDPR requirement.

Input always contains:

1. "REQUIREMENT" – text of GDPR requirement
2. "SYMBOLIC" - symbolic representation of GDPR requirement in Deontic Logic via ASP
3. "PREDICATES" - the symbolic_atom(s) repeated from a symbolic representation of the requirement, semicolon-separated.
4. "CLAUSE" - a single DPA segment.

VERY IMPORTANT: Carefully analyze the SYMBOLIC representation to understand the structure of the requirement. Pay special attention to:
- The main obligation or permission (what's inside &obligatory{} or &permitted{})
- The conditions that trigger the obligation (what follows the :- operator)
- How the predicates relate to each other
- Think step-by-step
- The text of requirement and the segment may be significantly different, you shoudl check if semantically they are mentioning similar things

Your task
- Decide which (if any) of the listed predicates the clause semantically entails.
- For every entailed predicate output the symbol followed by a period, one per line. Example:  ensure_confidentiality.
- If none are entailed, output exactly NO_FACTS.
- Produce nothing else: no prose, no JSON, no comments.

Think step-by-step in private, but reveal ONLY the final facts line(s) or NO_FACTS.
First, carefully analyze both the requirement and the DPA segment. Look for both explicit statements and implied relationships between entities (processor, controller, sub-processors, data subjects, etc.). Pay special attention to:

1. SEMANTIC EQUIVALENCES:
   - "Accountable" = "Liable" = "Responsible for" = "Answerable for"
   - "Ensure" = "Guarantee" = "Make certain" = "Make sure"
   - "Sub-processor" = "Subcontractor" = "Third party processor" = "Downstream processor"
   - "Personal data" = "Data" (when referring to information about individuals)
   - "Data subject" = "Individual" = "Person" (when referring to the person the data is about)

2. OBLIGATION INDICATORS:
   - "Shall" / "Must" / "Is required to" / "Is obligated to" / "Will" / "Is accountable for"
   - "In the same way as" / "Equally" / "To the same degree" (indicates equivalence of obligations)

3. IMPLICIT FACTS:
   - If a segment states "X is accountable to Y for Z," extract this as "X_liable_for_Z" or "X_responsible_for_Z"
   - If a segment discusses consequences or penalties, extract the underlying obligation

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
REQUIREMENT: The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller..
SYMBOLIC: &obligatory{-engage_sub_processor} :- -authorization(controller), role(processor).
PREDICATES: engage_sub_processor; role(processor); authorization(controller)
CLAUSE: Subject matter: The subject matter of the data processing under this DPA is the Personal Data.
Expected output: NO_FACTS
    """
    
    user_prompt = f"""
REQUIREMENT: {req_text}
SYMBOLIC: {req_symbolic}
PREDICATES: {'; '.join(req_predicates)}
CLAUSE: {segment_text}
"""
    
    # Get facts from LLM
    response = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
    
    # Process the response
    facts = {}
    
    # Filter out any explanatory text that might still be included
    # Keep only lines that match predicate patterns or "NO_FACTS"
    valid_lines = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Only keep lines that are either a single predicate or NO_FACTS
        if line == "NO_FACTS":
            valid_lines = ["NO_FACTS"]
            break
        # Check if the line is a valid predicate (either with or without a minus sign)
        if line.startswith('-'):
            predicate = line[1:].strip()
            # Remove any trailing period
            if predicate.endswith('.'):
                predicate = predicate[:-1]
            if predicate and ' ' not in predicate and ':' not in predicate:
                valid_lines.append(f"-{predicate}")
        else:
            # Remove any trailing period
            if line.endswith('.'):
                line = line[:-1]
            if line and ' ' not in line and ':' not in line:
                valid_lines.append(line)
    
    # If we filtered out everything but there was content, default to NO_FACTS
    if not valid_lines and response.strip():
        valid_lines = ["NO_FACTS"]
    
    # Process the valid lines
    if "NO_FACTS" in valid_lines:
        # Only add role facts based on the segment text
        if 'processor' in segment_text.lower():
            facts['role(processor)'] = True
        elif 'controller' in segment_text.lower():
            facts['role(controller)'] = True
    else:
        # Process normal fact extraction
        for line in valid_lines:
            line = line.strip()
            if line:
                if line.startswith('-'):
                    predicate = line[1:].strip()
                    facts[predicate] = False  # Negatively mentioned
                else:
                    facts[line] = True  # Positively mentioned
    
    # Add role facts if they're not already present
    if 'role(processor)' not in facts and 'role(controller)' not in facts:
        # Check if the segment mentions processor or controller
        if 'processor' in segment_text.lower():
            facts['role(processor)'] = True
        elif 'controller' in segment_text.lower():
            facts['role(controller)'] = True
    
    return facts

def generate_lp_file(req_symbolic, facts, req_predicates, req_text, segment_text):
    """Generate a complete LP file with the correct template structure."""
    # Clean up the symbolic representation to ensure valid syntax
    clean_symbolic = req_symbolic
    # Replace not(predicate) with -predicate for better compatibility
    clean_symbolic = re.sub(r'not\s*\(([^)]+)\)', r'-\1', clean_symbolic)
    
    # Add requirement and DPA segment text as comments
    lp_content = f"""% Requirement Text:
% {req_text}
%
% DPA Segment:
% {segment_text}
%
% 1. Normative layer
{clean_symbolic}

% 2. Facts extracted from DPA segment
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