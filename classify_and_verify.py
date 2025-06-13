#!/usr/bin/env python3
"""
classify_and_verify.py

Python script that processes DPA text segments using a two-step approach:
1. Classification Step: Determine which single GDPR requirement (if any) a segment is relevant to
2. Verification Step: Use symbolic reasoning to verify if that requirement is actually satisfied

This implements the RCV (Requirement Classification and Verification) approach.
"""

import os
import json
import argparse
import pandas as pd
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from ollama_client import OllamaClient


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process DPA segments using RCV approach"
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
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results and LP files"
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


def create_verification_prompt(segment_text: str, requirement_text: str, atoms: List[str]) -> str:
    """Create prompt for LLM verification step."""
    atoms_list = "\n".join([f"- {atom}" for atom in atoms])
    
    prompt = f"""You are analyzing a DPA segment to extract specific symbolic facts that prove a GDPR requirement is satisfied.

DPA Segment:
{segment_text}

GDPR Requirement:
{requirement_text}

Available Atoms (facts to look for):
{atoms_list}

Instructions:
1. Read the DPA segment carefully in the context of the specific GDPR requirement
2. Identify which of the available atoms are explicitly stated or clearly implied in the segment
3. Only include atoms that are actually present in the segment text
4. Return a semicolon-separated list of the relevant atom names (e.g., "role(processor);authorization(controller)")
5. If NO atoms from the list are found in the segment, return "NO_FACTS"

Important:
- Only return atom names that are clearly present in the segment
- Do not infer atoms that are not explicitly stated or clearly implied
- Use exact atom names from the list provided
- Separate multiple atoms with semicolons
- If no relevant atoms are found, return exactly "NO_FACTS"

Response:"""
    
    return prompt


def classify_segment(segment_text: str, requirements: Dict, llm_client: OllamaClient, model: str, verbose: bool = False) -> str:
    """Classify which requirement (if any) a segment is relevant to."""
    prompt = create_classification_prompt(segment_text, requirements)
    
    if verbose:
        print(f"Classification prompt:\n{prompt}\n")
    
    try:
        response = llm_client.generate(model, prompt, temperature=0.0)
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


def verify_segment(segment_text: str, requirement_text: str, atoms: List[str], llm_client: OllamaClient, model: str, verbose: bool = False) -> List[str]:
    """Extract symbolic facts from segment for verification."""
    prompt = create_verification_prompt(segment_text, requirement_text, atoms)
    
    if verbose:
        print(f"Verification prompt:\n{prompt}\n")
    
    try:
        response = llm_client.generate(model, prompt, temperature=0.0)
        facts_str = response.strip()
        
        if verbose:
            print(f"Verification response: {facts_str}")
        
        if facts_str == "NO_FACTS":
            return []
        else:
            # Parse semicolon-separated facts
            facts = [fact.strip() for fact in facts_str.split(';') if fact.strip()]
            # Validate facts are in atoms list
            valid_facts = [fact for fact in facts if fact in atoms]
            if verbose and len(valid_facts) != len(facts):
                print(f"Warning: Some facts were not in atoms list. Valid: {valid_facts}")
            return valid_facts
            
    except Exception as e:
        print(f"Error in verification: {e}")
        return []


def generate_static_verification_engine(requirements: Dict) -> str:
    """Generate the static verification engine with all requirement rules."""
    engine_rules = []
    
    for req_id, req_info in requirements.items():
        symbolic = req_info["symbolic"]
        atoms = req_info["atoms"]
        
        # Convert symbolic rule to ASP verification rule
        # Example: "&obligatory{process_data_on_documented_instructions} :- role(processor)."
        # Becomes: "requirement_satisfied(3) :- role(processor), process_data_on_documented_instructions."
        
        # Extract the obligation and body
        if "&obligatory{" in symbolic and "} :-" in symbolic:
            parts = symbolic.split("} :-")
            obligation_part = parts[0].replace("&obligatory{", "").strip()
            body_part = parts[1].strip().rstrip(".")
            
            # Create the verification rule
            rule = f"requirement_satisfied({req_id}) :- {body_part}, {obligation_part}."
            engine_rules.append(rule)
        elif "&obligatory{" in symbolic and ":-" not in symbolic:
            # Handle rules without conditions
            obligation_part = symbolic.replace("&obligatory{", "").replace("}", "").rstrip(".")
            rule = f"requirement_satisfied({req_id}) :- {obligation_part}."
            engine_rules.append(rule)
        else:
            # Fallback: create a basic rule structure
            obligation = f"obligation_{req_id}_satisfied"
            rule = f"requirement_satisfied({req_id}) :- {obligation}."
            engine_rules.append(rule)
    
    return "\n".join(engine_rules)


def generate_logic_program(classified_id: str, extracted_facts: List[str], static_engine: str) -> str:
    """Generate complete logic program for ASP solver."""
    program_parts = []
    
    # Part 1: Static Verification Engine
    program_parts.append("% Static Verification Engine")
    program_parts.append(static_engine)
    program_parts.append("")
    
    # Part 2: Dynamic Facts from LLM
    program_parts.append("% Dynamic Facts from LLM")
    
    # Classification fact
    if classified_id == "NONE":
        program_parts.append("classified_as(none).")
    else:
        program_parts.append(f"classified_as({classified_id}).")
    
    # Extracted facts
    for fact in extracted_facts:
        program_parts.append(f"{fact}.")
    
    program_parts.append("")
    
    # Part 3: Final Prediction Logic
    program_parts.append("% Final Prediction Logic")
    program_parts.append("final_prediction(R) :- requirement_satisfied(R).")
    program_parts.append("final_prediction(none) :- classified_as(R), R != none, not requirement_satisfied(R).")
    program_parts.append("final_prediction(none) :- classified_as(none).")
    program_parts.append("#show final_prediction/1.")
    
    return "\n".join(program_parts)


def solve_with_asp(lp_file_path: str, verbose: bool = False) -> str:
    """Solve logic program with ASP solver and return result."""
    try:
        # Try deolingo first (as used in the existing project)
        result = subprocess.run(
            ["deolingo", lp_file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"Deolingo error: {result.stderr}")
            return "none"
        
        output = result.stdout
        
        if verbose:
            print(f"Deolingo output: {output}")
        
        # Parse output for final_prediction
        for line in output.split('\n'):
            if 'final_prediction(' in line:
                # Extract prediction value
                import re
                match = re.search(r'final_prediction\(([^)]+)\)', line)
                if match:
                    prediction = match.group(1)
                    return prediction
        
        # Default to none if no prediction found
        return "none"
        
    except subprocess.TimeoutExpired:
        if verbose:
            print("Deolingo timeout")
        return "none"
    except FileNotFoundError:
        print("Error: 'deolingo' command not found. Please install deolingo ASP solver.")
        print("Install with: pip3 install git+https://github.com/ovidiomanteiga/deolingo.git@main")
        return "none"
    except Exception as e:
        if verbose:
            print(f"Error running deolingo: {e}")
        return "none"


def process_dpa_segments(segments_df: pd.DataFrame, requirements: Dict, llm_client: OllamaClient, 
                        model: str, output_dir: str, target_dpa: str, verbose: bool = False) -> List[Dict]:
    """Process all DPA segments using RCV approach."""
    results = []
    
    # Generate static verification engine
    static_engine = generate_static_verification_engine(requirements)
    
    # Create output directories
    lp_files_dir = os.path.join(output_dir, "lp_files", target_dpa.replace(" ", "_"))
    os.makedirs(lp_files_dir, exist_ok=True)
    
    # Process each segment
    for idx, row in tqdm(segments_df.iterrows(), total=len(segments_df), desc="Processing segments"):
        segment_id = row["ID"]
        segment_text = row["Sentence"]
        
        # Extract ground truth if available
        ground_truth = "unknown"
        if "target" in row and pd.notna(row["target"]):
            ground_truth = str(row["target"])
        
        if verbose:
            print(f"\nProcessing segment {segment_id}: {segment_text[:100]}...")
        
        # Step 1: Classification
        classified_id = classify_segment(segment_text, requirements, llm_client, model, verbose)
        
        # Step 2: Verification (only if classified)
        extracted_facts = []
        if classified_id != "NONE":
            requirement_info = requirements[classified_id]
            requirement_text = requirement_info["text"]
            atoms = requirement_info["atoms"]
            
            extracted_facts = verify_segment(segment_text, requirement_text, atoms, llm_client, model, verbose)
        
        # Step 3: Generate Logic Program
        lp_content = generate_logic_program(classified_id, extracted_facts, static_engine)
        
        # Save LP file
        lp_file_path = os.path.join(lp_files_dir, f"segment_{segment_id}.lp")
        with open(lp_file_path, 'w') as f:
            f.write(lp_content)
        
        # Step 4: Solve with ASP
        final_prediction = solve_with_asp(lp_file_path, verbose)
        
        # Store results
        result = {
            "Segment_ID": segment_id,
            "DPA": target_dpa,
            "Segment_Text": segment_text,
            "LLM_Classification": classified_id,
            "Extracted_Facts": ";".join(extracted_facts) if extracted_facts else "NO_FACTS",
            "Final_Prediction": final_prediction,
            "Ground_Truth": ground_truth
        }
        results.append(result)
        
        if verbose:
            print(f"Result: Classification={classified_id}, Final={final_prediction}, Ground Truth={ground_truth}")
    
    return results


def main():
    """Main function."""
    args = parse_arguments()
    
    print("========== DPA Completeness Checker - RCV Approach ==========")
    print(f"Target DPA: {args.target_dpa}")
    print(f"Model: {args.model}")
    print(f"Output Directory: {args.output_dir}")
    print("=============================================================")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Ollama client
    print("Initializing Ollama client...")
    llm_client = OllamaClient()
    
    if not llm_client.check_health():
        print("Error: Ollama server is not running. Please start it first.")
        sys.exit(1)
    
    # Load requirements
    print(f"Loading requirements from: {args.requirements}")
    requirements = load_requirements(args.requirements)
    print(f"Loaded {len(requirements)} requirements")
    
    # Load DPA segments
    print(f"Loading DPA segments from: {args.dpa_segments}")
    segments_df = load_dpa_segments(args.dpa_segments, args.target_dpa, args.max_segments)
    print(f"Loaded {len(segments_df)} segments for DPA: {args.target_dpa}")
    
    # Process segments
    print("Processing segments with RCV approach...")
    results = process_dpa_segments(
        segments_df, requirements, llm_client, args.model, 
        args.output_dir, args.target_dpa, args.verbose
    )
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(args.output_dir, f"rcv_results_{args.target_dpa.replace(' ', '_')}.csv")
    results_df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Processed {len(results)} segments")
    
    # Print summary statistics
    classification_counts = results_df['LLM_Classification'].value_counts()
    print(f"\nClassification Summary:")
    for class_id, count in classification_counts.items():
        print(f"  {class_id}: {count}")
    
    prediction_counts = results_df['Final_Prediction'].value_counts()
    print(f"\nFinal Prediction Summary:")
    for pred, count in prediction_counts.items():
        print(f"  {pred}: {count}")


if __name__ == "__main__":
    main() 