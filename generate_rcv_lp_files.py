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


def create_classification_prompt(segment_text: str, requirements: Dict) -> str:
    """Create prompt for LLM classification step."""
    req_list = []
    for req_id, req_info in requirements.items():
        req_list.append(f"{req_id}: {req_info['text']}")
    
    prompt = f"""You are analyzing a DPA (Data Processing Agreement) segment to determine which single GDPR requirement it is most relevant to.

DPA Segment:
{segment_text}

Available GDPR Requirements:
{chr(10).join(req_list)}

Instructions:
1. Read the DPA segment carefully
2. Determine if it is relevant to any of the GDPR requirements listed above
3. If it is relevant, identify the SINGLE most relevant requirement ID (e.g., "3")
4. If it is NOT relevant to any requirement (e.g., it's a heading, boilerplate, or administrative text), return "NONE"

Important:
- Return ONLY the requirement ID (e.g., "3") or "NONE"
- Do not include any explanation or additional text
- Choose only ONE requirement ID, even if multiple might apply
- If unsure, return "NONE"

Response:"""
    
    return prompt





def classify_segment(segment_text: str, requirements: Dict, llm_client: OllamaClient, model: str, verbose: bool = False) -> str:
    """Classify which requirement (if any) a segment is relevant to."""
    system_prompt = """You are a legal expert specializing in GDPR compliance analysis. Your task is to classify DPA segments according to which GDPR requirement they are most relevant to.

You will be given:
1. A DPA segment (text from a Data Processing Agreement)
2. A list of GDPR requirements with their IDs and descriptions

Your task:
- Determine which single GDPR requirement (if any) the DPA segment is most relevant to
- Output ONLY the requirement ID (e.g., "3", "7", "15") 
- If the segment is not relevant to any requirement, output "NONE"
- Do not provide explanations or multiple IDs

Focus on:
- Processor obligations and responsibilities
- Data protection measures and safeguards
- Legal compliance requirements
- Contractual obligations between controller and processor

Ignore:
- Administrative text (definitions, contact info, etc.)
- General business terms unrelated to data protection
- Purely commercial clauses"""

    user_prompt = create_classification_prompt(segment_text, requirements)
    
    if verbose:
        print(f"Classification prompt:\n{user_prompt}\n")
    
    try:
        response = llm_client.generate(user_prompt, model_name=model, system_prompt=system_prompt)
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
    """Extract facts from a DPA segment using the LLM, matching existing format."""
    system_prompt = """You are a legal-text expert that extracts facts from Data-Processing-Agreement (DPA) segments based on semantic and contextual similarity with GDPR regulatory requirements.

Input always contains:
1. "REQUIREMENT" – text of the GDPR requirement
2. "PREDICATES" – ASP atoms from the requirement (semicolon-separated)
3. "CLAUSE" – one DPA segment

TASK:
Decide which (if any) predicates are explicitly fully mentioned in the CLAUSE and output them separated by semicolon

INSTRUCTIONS:
1) Output a predicate from symbolic rule's body only if the CLAUSE explicitly and fully mentions the same concept this predicate mentions in the REQUIREMENT.
2) Output a predicate from symbolic rule's head only if the CLAUSE describes a rule for a processor and this rule is semantically the same as the REQUIREMENT
3) If no predicated are entailed, output exactly NO_FACTS
4) If the CLAUSE explicitly violates a predicate, output it prefixed by - (e.g. -encrypt_data)
5) Output ONLY extracted predicates or NO_FACTS, do not output explanation or something else."""

    user_prompt = f""" REQUIREMENT: {req_text} PREDICATES: {'; '.join(req_predicates)} CLAUSE: {segment_text}"""
    
    try:
        response = llm_client.generate(user_prompt, model_name=model, system_prompt=system_prompt)
        response = response.strip()
        
        # Parse the response
        if response == "NO_FACTS":
            return {}
        facts = {}
        for pred in response.split(';'):
            pred = pred.strip()
            if pred.startswith('-'):
                facts[pred[1:]] = False
            else:
                facts[pred] = True
        
        # CRITICAL CHECK: If role(processor) is not among the extracted facts, 
        # treat this segment as NO_FACTS since GDPR requirements are about processor obligations.
        if "role(processor)" not in facts:
            return {}
        
        return facts
        
    except Exception as e:
        print(f"Error in fact extraction: {e}")
        return {}


def process_dpa_segments(segments_df: pd.DataFrame, requirements: Dict, llm_client: OllamaClient, 
                        model: str, output_dir: str, verbose: bool = False) -> None:
    """Process all DPA segments using RCV approach and generate LP files."""
    
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
        
        if classified_id == "NONE":
            # Create a single LP file indicating no classification
            lp_content = f"""% Requirement Text:
% Administrative or non-relevant segment
%
% DPA Segment:
% {segment_text}
%

% No relevant GDPR requirement found for this segment
status(not_mentioned) :- true.

#show status/1.
"""
            lp_file_path = os.path.join(output_dir, f"segment_{segment_id}.lp")
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
        else:
            # Step 2: Verification for the classified requirement
            requirement_info = requirements[classified_id]
            req_text = requirement_info["text"]
            req_symbolic = requirement_info["symbolic"]
            req_predicates = requirement_info["atoms"]
            
            # Extract facts
            facts = extract_facts_from_dpa(segment_text, req_text, req_symbolic, req_predicates, 
                                         llm_client, model)
            
            # Generate LP file content
            lp_content = generate_lp_file(segment_text, req_text, req_symbolic, facts, req_predicates)
            
            # Save LP file
            lp_file_path = os.path.join(output_dir, f"segment_{segment_id}.lp")
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
        
        if verbose:
            print(f"Generated LP file: {lp_file_path}")
            if classified_id != "NONE":
                print(f"Classification: {classified_id}, Facts: {facts}")


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