# generate_focused_lp_files.py
import os
import json
import argparse
import re
from tqdm import tqdm
from models.llm import LlamaModel
from config.llm_config import LlamaConfig
from extract_facts import extract_facts_from_symbolic

def create_lp_file_content(req_symbolic, dpa_symbolic, semantic_rule=None, req_id="req"):
    """Create a syntactically correct LP file for deolingo with extracted facts."""
    # Normalize symbols for better compatibility
    req_symbolic = req_symbolic.replace(' | ', '; ')
    dpa_symbolic = dpa_symbolic.replace(' | ', '; ')
    
    # Extract facts from symbolic representations
    req_facts = extract_facts_from_symbolic(req_symbolic)
    dpa_facts = extract_facts_from_symbolic(dpa_symbolic)
    
    # Combine facts and remove duplicates
    all_facts = req_facts.union(dpa_facts)
    
    # Extract additional facts from semantic rule if provided
    if semantic_rule:
        semantic_facts = extract_facts_from_symbolic(semantic_rule)
        all_facts = all_facts.union(semantic_facts)
    
    # Start with definitions
    lp_content = f"""% Requirement symbolic representation
{req_symbolic}

% DPA segment symbolic representation
{dpa_symbolic}
"""
    
    # Add semantic rule if provided
    if semantic_rule:
        lp_content += f"""
% Semantic connection rule between requirement and DPA
{semantic_rule}
"""
    
    # Add all extracted facts
    if all_facts:
        lp_content += """
% Facts - Extracted from symbolic representations
"""
        for fact in sorted(all_facts):
            lp_content += f"{fact}\n"
    
    # Extract predicates from deontic operators for satisfaction rules
    head_predicates = []
    for line in req_symbolic.split('\n'):
        if '&obligatory{' in line:
            match = re.search(r'&obligatory{([^}]+)}', line)
            if match:
                pred = match.group(1).strip()
                if pred not in head_predicates:
                    head_predicates.append(pred)
    
    # Add satisfaction, violation, and not_mentioned definitions
    lp_content += """
% Satisfaction, violation, and not_mentioned definitions
"""
    
    # Create satisfaction rules for each predicate in requirement obligations
    if head_predicates:
        satisfaction_conditions = " | ".join([f"({pred})" for pred in head_predicates])
        lp_content += f"satisfies({req_id}) :- {satisfaction_conditions}.\n"
    else:
        # Default satisfaction rule if no predicates found
        lp_content += f"satisfies({req_id}) :- &obligatory{{X}}, X.\n"
    
    # Add violation and not_mentioned rules
    lp_content += f"""
violates({req_id}) :- &obligatory{{X}}, &forbidden{{X}}.
not_mentioned({req_id}) :- not satisfies({req_id}), not violates({req_id}).

% Show directives
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.
"""
    
    return lp_content

def generate_focused_lp_files(requirements_file, dpa_segments_file, model_path, output_dir, target_req_id="5", segment_limit=30):
    """Generate LP files focused on requirement #5 and first 30 segments."""
    # Initialize LLM
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    
    # Load requirements
    with open(requirements_file, 'r') as f:
        requirements_data = json.load(f)
    
    # Find requirement #5 (security measures)
    target_req = None
    for req_text, req_symbolic in requirements_data.items():
        if f"The processor shall take all measures required pursuant to Article 32" in req_text:
            target_req = {
                "text": req_text,
                "symbolic": req_symbolic
            }
            break
    
    if not target_req:
        print(f"Error: Target requirement (Article 32) not found in requirements file")
        return
    
    # Load DPA segments
    with open(dpa_segments_file, 'r') as f:
        dpa_data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for LP files
    lp_dir = os.path.join(output_dir, f"req_{target_req_id}")
    os.makedirs(lp_dir, exist_ok=True)
    
    # Generate semantic rules and LP files for the first 30 segments
    print(f"Processing up to {segment_limit} DPA segments for requirement #{target_req_id}")
    
    # Get first 30 segments (or fewer if not available)
    segments = list(dpa_data["segments"].items())[:segment_limit]
    
    # Track results
    results = {
        "req_id": target_req_id,
        "dpa_id": dpa_data.get("dpa_id", "unknown"),
        "segments": {}
    }
    
    for segment_id, segment_info in tqdm(segments):
        # Get segment info
        dpa_text = segment_info["text"]
        dpa_symbolic = segment_info.get("symbolic", "")
        
        # Generate semantic rule
        semantic_rule = generate_semantic_rule(
            llm_model,
            target_req["text"],
            target_req["symbolic"],
            dpa_text,
            dpa_symbolic
        )
        
        # Track if semantic connection exists
        results["segments"][segment_id] = {
            "has_semantic_connection": semantic_rule is not None
        }
        
        # Create LP file
        lp_file_path = os.path.join(lp_dir, f"segment_{segment_id}.lp")
        
        lp_content = create_lp_file_content(
            target_req["symbolic"],
            dpa_symbolic,
            semantic_rule,
            req_id="req"
        )
        
        with open(lp_file_path, 'w') as f:
            f.write(lp_content)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"req_{target_req_id}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated LP files for requirement #{target_req_id} in {lp_dir}")
    print(f"Summary saved to {summary_file}")
    
    return lp_dir

def main():
    parser = argparse.ArgumentParser(description="Generate focused LP files for DPA compliance checking")
    parser.add_argument("--requirements", type=str, required=True, 
                       help="Path to requirements JSON file")
    parser.add_argument("--dpa_segments", type=str, required=True,
                       help="Path to DPA segments JSON file")  
    parser.add_argument("--model", type=str, required=True,
                       help="Path to LLM model")
    parser.add_argument("--output", type=str, default="results/focused",
                       help="Path to output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate focused LP files
    generate_focused_lp_files(
        args.requirements,
        args.dpa_segments,
        args.model,
        args.output
    )

if __name__ == "__main__":
    main()