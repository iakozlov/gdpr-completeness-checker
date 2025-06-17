#!/usr/bin/env python3
"""
generate_rcv_lp_files.py

Python script that generates LP files using the RCV (Requirement Classification and Verification) approach.
This script follows the same pattern as generate_lp_files.py but implements the two-step RCV logic:
1. Classification Step: Determine which single GDPR requirement (if any) a segment is relevant to
2. Verification Step: Extract symbolic facts specific to that requirement

This script only generates .lp files - the solver is called separately by the shell script.
"""

import os
import json
import argparse
import pandas as pd
import re
from typing import Dict, List, Optional
from tqdm import tqdm
from ollama_client import OllamaClient


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RCV LP files for DPA segments"
    )
    parser.add_argument(
        "--requirements",
        type=str,
        required=True,
        help="Path to requirements JSON file"
    )
    parser.add_argument(
        "--dpa_segments",
        type=str,
        required=True,
        help="Path to DPA segments CSV file"
    )
    parser.add_argument(
        "--target_dpa",
        type=str,
        required=True,
        help="Target DPA name to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for LP files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.3:70b",
        help="Ollama model to use (default: llama3.3:70b)"
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=0,
        help="Maximum number of segments to process (0 means all, default: 0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def load_requirements(file_path: str) -> Dict:
    """Load requirements from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_dpa_segments(file_path: str, target_dpa: str, max_segments: int = 0) -> pd.DataFrame:
    """Load and filter DPA segments."""
    df = pd.read_csv(file_path)
    
    # Filter for target DPA
    df_filtered = df[df['DPA'] == target_dpa].copy()
    
    if df_filtered.empty:
        raise ValueError(f"No segments found for DPA: {target_dpa}")
    
    # Apply segment limit if specified
    if max_segments > 0:
        df_filtered = df_filtered.head(max_segments)
    
    # Reset index for consistent iteration
    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered


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


def classify_segment(segment_text: str, requirements: Dict, llm_client: OllamaClient, model: str, verbose: bool = False) -> str:
    """Classify which requirement (if any) a segment is relevant to."""
    # Build requirements list for system prompt
    req_list = []
    for req_id, req_info in requirements.items():
        req_list.append(f"{req_id}: {req_info['text']}")
    
    system_prompt = f"""You are a legal expert specializing in GDPR compliance analysis. Your task is to classify DPA segments according to which GDPR requirement they are most relevant to.

AVAILABLE GDPR REQUIREMENTS:
{chr(10).join(req_list)}

TASK:
Classify the given DPA segment to determine which single GDPR requirement it is most relevant to. Return ONLY the requirement ID (e.g., "3") or "NONE" if not relevant to any requirement.

CLASSIFICATION RULES:
- Focus on processor obligations and responsibilities
- Identify data protection measures and safeguards  
- Look for legal compliance requirements
- Recognize contractual obligations between controller and processor
- Ignore administrative text, definitions, general business terms
- If unsure, return "NONE"
- Choose only ONE requirement ID, even if multiple might apply

EXAMPLES:

Input: "The processor shall not subcontract any of its processing operations performed on behalf of the controller under the Clauses without the prior written consent of the controller."
Output: 1

Input: "processor shall notify controller with at least ten (10) days' prior notice before authorizing any new Sub-Processors to access controller's Personal Data;"
Output: 2

Input: "processor will process controller Data only in accordance with Documented Instructions."
Output: 3

Input: "In such cases, the processor shall inform the controller of the legal requirement before processing, unless that law prohibits such information on important grounds of public interest."
Output: 4

Input: "processor imposes appropriate contractual obligations upon its personnel, including relevant obligations regarding confidentiality, data protection and data security."
Output: 5

Input: "Guests and visitors to processor buildings must register their names at reception and must be accompanied by authorized processor personnel."
Output: 6

Input: "In the event any such request is made directly to processor, processor shall notify controller in writing of such request promptly upon receipt thereof."
Output: 7

Input: "The processor may add to, amend, or replace the specific security measures for security reasons and shall notify the controller in writing where it has done so."
Output: 8

Input: "Prior consultation with the Supervisory Authority;"
Output: 9

Input: "If the Controller has an obligation to notify or report in the event of a security incident, the Processor is obliged to support the Controller at the Controllers expense."
Output: 10

Input: "Processor shall assist Controller with any notifications to data subjects and/or authorities if requested by Controller."
Output: 11

Input: "The processor shall work with the controller to carry out a risk assessment and allow them to oversee and assess any corrective action."
Output: 12

Input: "processor will delete controller Data when requested by controller by using the Service controls provided for this purpose by processor."
Output: 13

Input: "processor shall inform controller if, in the opinion of processor, an instruction infringes Applicable Data Protection Law."
Output: 14

Input: "processor will make available to controller all information necessary to demonstrate compliance with the obligations of Data Processors laid down in Article 28 of GDPR."
Output: 15

Input: "Processor shall grant Controller access to all information required in order to verify that the obligations set out in the DPA are complied with."
Output: 16

Input: "Processor will ensure that Sub-processors are bound by written agreements that require them to provide at least the level of data protection required of Processor by these GDPR Terms."
Output: 17

Input: "processor shall remain fully liable to controller for the performance of the Sub-Processor's obligations."
Output: 18

Input: "processor has implemented and will maintain appropriate technical and organizational security measures for the Processing of Personal Data."
Output: 19

Input: "This DPA applies when controller Data is processed by processor."
Output: NONE

Input: "The subject matter of the data processing under this DPA is controller Data."
Output: NONE

Input: "Additional instructions outside the scope of the Documented Instructions (if any) require prior written agreement between processor"
Output: NONE"""

    user_prompt = segment_text
    
    if verbose:
        print(f"Classification prompt:\n{user_prompt}\n")
    
    try:
        response = llm_client.generate(user_prompt, model_name=model, system_prompt=system_prompt)
        # Filter out thinking sections
        response = filter_think_sections(response)
        classified_id = response.strip()
        
        if verbose:
            print(f"Classification response: {classified_id}")
        
        # Validate response
        if classified_id == "NONE":
            return "NONE"
        elif classified_id in requirements:
            return classified_id
        else:
            if verbose:
                print(f"Warning: Invalid classification response '{classified_id}', returning 'NONE'")
            return "NONE"
            
    except Exception as e:
        print(f"Error in classification: {e}")
        return "NONE"


def extract_body_atoms(symbolic_rule):
    """Extract atoms from the body of an ASP rule."""
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


def generate_lp_file(segment_text: str, req_text: str, req_symbolic: str, facts: Dict, req_predicates: List[str]) -> str:
    """Generate LP file content matching the existing format."""
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


def extract_facts_from_dpa(segment_text: str, req_text: str, req_symbolic: str, req_predicates: List[str], 
                           llm_client: OllamaClient, model: str) -> Dict:
    """Extract facts from a DPA segment using the LLM."""
    system_prompt = """You are a legal-text expert that extracts facts from Data-Processing-Agreement (DPA) segments based on semantic and contextual similarity with GDPR regulatory requirements.

Input always contains:
1. "REQUIREMENT" – text of the GDPR requirement
2. "SYMBOLIC" – symbolic representation of the requirement in deontic logic via Answer Set Programming (ASP)
3. "PREDICATES" – ASP atoms from the requirement (semicolon-separated)
4. "CLAUSE" – one DPA segment

TASK:
Decide which (if any) predicates are explicitly fully mentioned in the CLAUSE and output them separated by semicolon

INSTRUCTIONS:
1) Output a predicate from symbolic rule's body only if the CLAUSE explicitly and fully mentions the same concept this predicate mentions in the REQUIREMENT.
2) Output a predicate from symbolic rule's head only if the CLAUSE describes a rule for a processor and this rule is semantically the same as the REQUIREMENT
3) If no predicated are entailed, output exactly NO_FACTS
4) If the CLAUSE explicitly violates a predicate, output it prefixed by - (e.g. -encrypt_data)
5) Output ONLY extracted predicates or NO_FACTS, do not output explanation or something else.

Examples:
Example 1:
REQUIREMENT: The processor shall ensure that persons authorized to process personal data have committed themselves to confidentiality or are under an appropriate statutory obligation of confidentiality.
SYMBOLIC: &obligatory{ensure_confidentiality_commitment} :- role(processor).
PREDICATES: ensure_confidentiality_commitment; role(processor)
CLAUSE: The Processor shall ensure that every employee authorized to process Customer Personal Data is subject to a contractual duty of confidentiality.
Expected output: ensure_confidentiality_commitment; role(processor)

Example 2:
REQUIREMENT: The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller.
SYMBOLIC: &obligatory{-engage_sub_processor} :- role(processor), not authorization(controller).
PREDICATES: engage_sub_processor; role(processor); authorization(controller)
CLAUSE: Where processor authorises any sub-processor as described in Section 6.1
Expected output: role(processor)

Example 3:
REQUIREMENT: The processor must encrypt all the data collected from customers.
SYMBOLIC: &obligatory{encrypt_collected_data} :- role(processor)
PREDICATES: encrypt_collected_data; role(processor)
CLAUSE: The processor will store customer's data in raw format.
Expected output: -encrypt_collected_data; role(processor)

Example 4:
REQUIREMENT: The processor shall process personal data only on documented instructions from the controller.
SYMBOLIC: &obligatory{process_on_documented_instructions} :- role(processor).
PREDICATES: process_on_documented_instructions; role(processor)
CLAUSE: This Data Processing Addendum ("DPA") supplements the processor controller Agreement available at as updated from time to time between controller and processor, or other agreement between controller and processor governing controller's use of the Service Offerings.
Expected output: NO_FACTS."""

    user_prompt = f""" REQUIREMENT: {req_text} SYMBOLIC: {req_symbolic} PREDICATES: {'; '.join(req_predicates)} CLAUSE: {segment_text}"""
    
    try:
        response = llm_client.generate(user_prompt, model_name=model, system_prompt=system_prompt)
        # Filter out thinking sections
        response = filter_think_sections(response)
        response = response.strip()
        
        if response == "NO_FACTS":
            return {}
            
        # Parse the response into a dictionary of facts
        facts = {}
        for pred in response.split(';'):
            pred = pred.strip()
            if pred and pred != "NO_FACTS" and pred != "NO_FACTS.": 
                if pred.startswith('-'):
                    facts[pred[1:]] = False
                else:
                    facts[pred] = True
        return facts
        
    except Exception as e:
        print(f"Error in fact extraction: {e}")
        return {}


def process_dpa_segments(segments_df: pd.DataFrame, requirements: Dict, llm_client: OllamaClient, 
                        model: str, output_dir: str, verbose: bool = False) -> None:
    """Process all DPA segments using RCV approach and generate LP files compatible with existing evaluation."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each segment
    for idx, row in tqdm(segments_df.iterrows(), total=len(segments_df), desc="Processing segments"):
        segment_id = row["ID"]
        segment_text = row["Sentence"]
        
        if verbose:
            print(f"\nProcessing segment {segment_id}: {segment_text[:100]}...")
        
        # Step 1: Classification
        classified_id = classify_segment(segment_text, requirements, llm_client, model, verbose)
        
        # Generate LP files for all requirements (to maintain compatibility with evaluation)
        for req_id, requirement_info in requirements.items():
            req_text = requirement_info["text"]
            req_symbolic = requirement_info["symbolic"]
            req_predicates = requirement_info["atoms"]
            
            # Create requirement directory
            req_dir = os.path.join(output_dir, f"req_{req_id}")
            os.makedirs(req_dir, exist_ok=True)
            
            if req_id == classified_id:
                # Step 2: Verification for the classified requirement
                facts = extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, 
                                             llm_client, model)
                
                # Generate LP file content
                lp_content = generate_lp_file(segment_text, req_text, req_symbolic, facts, req_predicates)
            else:
                # For non-classified requirements, generate "not_mentioned" LP file
                lp_content = f"""% Requirement Text:
% {req_text}
%
% DPA Segment:
% {segment_text}
%

% RCV Classification: This segment was not classified as relevant to this requirement
% Classified as: {classified_id if classified_id != "NONE" else "Administrative/Non-relevant"}
status(not_mentioned) :- true.

#show status/1.
"""
            
            # Save LP file
            lp_file_path = os.path.join(req_dir, f"segment_{segment_id}.lp")
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
        
        if verbose:
            print(f"Generated LP files for segment {segment_id}")
            print(f"Classification: {classified_id}")
            if classified_id != "NONE":
                print(f"Verified requirement {classified_id}")


def main():
    """Main function."""
    args = parse_arguments()
    
    print("========== RCV LP File Generator ==========")
    print(f"Target DPA: {args.target_dpa}")
    print(f"Model: {args.model}")
    print(f"Output Directory: {args.output}")
    print("==========================================")
    
    # Initialize Ollama client
    print("Initializing Ollama client...")
    llm_client = OllamaClient()
    
    if not llm_client.check_health():
        print("Error: Ollama server is not running. Please start it first.")
        return 1
    
    # Load requirements
    print(f"Loading requirements from: {args.requirements}")
    requirements = load_requirements(args.requirements)
    print(f"Loaded {len(requirements)} requirements")
    
    # Load DPA segments
    print(f"Loading DPA segments from: {args.dpa_segments}")
    segments_df = load_dpa_segments(args.dpa_segments, args.target_dpa, args.max_segments)
    print(f"Loaded {len(segments_df)} segments for DPA: {args.target_dpa}")
    
    # Process segments and generate LP files
    print("Processing segments with RCV approach...")
    process_dpa_segments(
        segments_df, requirements, llm_client, args.model, 
        args.output, args.verbose
    )
    
    print(f"\nLP file generation completed!")
    print(f"Generated {len(segments_df)} LP files in: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 