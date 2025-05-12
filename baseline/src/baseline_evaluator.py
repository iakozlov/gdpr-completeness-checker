import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging to suppress debug messages
logging.basicConfig(level=logging.INFO)
# Set loguru level before importing
os.environ["LOGURU_LEVEL"] = "INFO"

# Add project root to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.llm import LlamaModel
from config.llm_config import LlamaConfig

def load_requirements(requirements_file: str) -> Dict[str, str]:
    """Load requirements from file."""
    requirements = {}
    with open(requirements_file, 'r') as f:
        for line in f:
            if line.strip():
                req_id, req_text = line.split('.', 1)
                requirements[req_id.strip()] = req_text.strip()
    return requirements

def load_dpa_segments(dpa_csv: str, target_dpa: str, max_segments: int) -> List[str]:
    """Load DPA segments for the target DPA."""
    df = pd.read_csv(dpa_csv)
    # Filter by DPA name and get the Sentence column
    dpa_segments = df[df['DPA'] == target_dpa]['Sentence'].tolist()
    
    # If max_segments is 0, return all segments
    if max_segments == 0:
        return dpa_segments
    
    # Otherwise, return up to max_segments
    return dpa_segments[:max_segments]

def create_prompt(requirement: str, segment: str) -> str:
    """Create a prompt for the LLM to evaluate requirement satisfaction."""
    return f"""You are an expert in evaluating Data Processing Agreements (DPAs) against GDPR requirements.
Given a DPA segment and a requirement, determine if the segment satisfies the requirement.

Requirement: {requirement}
DPA Segment: {segment}

Does this DPA segment satisfy the requirement? Answer with only 'yes' or 'no'.
If the segment is not relevant to the requirement, answer 'no'.
If the segment partially satisfies the requirement, answer 'no'.
Only answer 'yes' if the segment fully satisfies the requirement.

Answer:"""

def evaluate_segment(llm: LlamaModel, requirement: str, segment: str) -> bool:
    """Evaluate if a segment satisfies a requirement using LLM."""
    prompt = create_prompt(requirement, segment)
    response = llm.generate_response(prompt, max_tokens=10)
    # Check first word for yes/no
    answer = response.split()[0].lower() if response else "no"
    return answer == 'yes'

def main():
    parser = argparse.ArgumentParser(description='Baseline DPA completeness evaluator')
    parser.add_argument('--requirements', required=True, help='Path to requirements file')
    parser.add_argument('--dpa', required=True, help='Path to DPA CSV file')
    parser.add_argument('--model', required=True, help='Path to LLM model file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--target_dpa', required=True, help='Target DPA name')
    parser.add_argument('--max_segments', type=int, default=20, help='Maximum number of segments to process (0 means all)')
    parser.add_argument('--req_ids', required=True, help='Comma-separated list of requirement IDs or "all" for all requirements')
    
    args = parser.parse_args()
    
    # Load requirements
    all_requirements = load_requirements(args.requirements)
    
    # Handle req_ids parameter
    if args.req_ids.lower() == 'all':
        req_ids = list(all_requirements.keys())
        print(f"Processing all {len(req_ids)} requirements")
    else:
        req_ids = [req_id.strip() for req_id in args.req_ids.split(',')]
        print(f"Processing {len(req_ids)} specified requirements: {', '.join(req_ids)}")
    
    # Load DPA segments
    segments = load_dpa_segments(args.dpa, args.target_dpa, args.max_segments)
    print(f"Processing {len(segments)} segments for DPA: {args.target_dpa}")
    
    # Initialize LLM using the project's LlamaModel
    config = LlamaConfig(model_path=args.model)
    llm = LlamaModel(config)
    
    # Evaluate each requirement against each segment
    results = {
        'dpa_name': args.target_dpa,
        'requirements': {},
        'metrics': {
            'total_requirements': len(req_ids),
            'satisfied_requirements': 0,
            'total_segments': len(segments),
            'satisfaction_matrix': []
        }
    }
    
    # Create satisfaction matrix
    satisfaction_matrix = []
    
    # Use tqdm for progress tracking
    print(f"Evaluating {len(req_ids)} requirements against {len(segments)} segments...")
    for req_id in tqdm(req_ids, desc="Processing requirements"):
        req_text = all_requirements[req_id]
        req_results = {
            'requirement_id': req_id,
            'requirement_text': req_text,
            'satisfied': False,
            'satisfying_segments': []
        }
        
        for i, segment in enumerate(tqdm(segments, desc=f"Processing segments for req {req_id}", leave=False)):
            is_satisfied = evaluate_segment(llm, req_text, segment)
            if is_satisfied:
                req_results['satisfying_segments'].append(i)
        
        req_results['satisfied'] = len(req_results['satisfying_segments']) > 0
        if req_results['satisfied']:
            results['metrics']['satisfied_requirements'] += 1
        
        results['requirements'][req_id] = req_results
        satisfaction_matrix.append([1 if i in req_results['satisfying_segments'] else 0 for i in range(len(segments))])
    
    results['metrics']['satisfaction_matrix'] = satisfaction_matrix
    results['metrics']['completeness_score'] = results['metrics']['satisfied_requirements'] / results['metrics']['total_requirements']
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main() 