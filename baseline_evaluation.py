# baseline_evaluation.py
import os
import json
import argparse
import pandas as pd
import random
import re
from tqdm import tqdm
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

def load_requirements():
    """Load the ground truth requirements (1-19)."""
    requirements = {}
    with open("data/requirements/ground_truth_requirements.txt", 'r') as f:
        content = f.read().strip()
        
    # Split by requirement numbers and clean up
    req_lines = content.split('\n')
    current_req = None
    current_text = []
    
    for line in req_lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a number followed by a period
        if re.match(r'^\d+\.', line):
            # Save previous requirement if exists
            if current_req is not None:
                requirements[current_req] = ' '.join(current_text).strip()
            
            # Extract requirement number and text
            parts = line.split('.', 1)
            current_req = int(parts[0])
            current_text = [parts[1].strip()]
        else:
            # Continue previous requirement text
            current_text.append(line)
    
    # Save the last requirement
    if current_req is not None:
        requirements[current_req] = ' '.join(current_text).strip()
    
    return requirements

def map_requirement_labels(label):
    """Map R10-R29 labels to requirement numbers 1-19."""
    if label.startswith('R'):
        try:
            r_num = int(label[1:])
            if 10 <= r_num <= 29:
                return r_num - 9  # R10 -> 1, R11 -> 2, ..., R29 -> 20
            # Handle cases like R3, R4, etc. that don't fit the R10-R29 pattern
            return None
        except ValueError:
            return None
    return None

def req_number_to_r_label(req_number: int) -> str:
    """Map requirement number back to R-label for consistency with evaluate_completeness.py"""
    # Map requirement 1-19 back to R10-R29 (excluding R14)
    mapping = {
        1: "10", 2: "11", 3: "12", 4: "13", 5: "15", 6: "16", 7: "17", 8: "18", 9: "19",
        10: "20", 11: "21", 12: "22", 13: "23", 14: "24", 15: "25", 16: "26", 17: "27", 18: "28", 19: "29"
    }
    return mapping.get(req_number, str(req_number))

def classify_dpa_segment(requirement_text, dpa_segment, llm_model, model_name="llama3.3:70b"):
    """
    Use LLM to classify whether a DPA segment satisfies, violates, or doesn't mention a requirement.
    
    Args:
        requirement_text: The text of the GDPR requirement
        dpa_segment: The text of the DPA segment
        llm_model: The LLM model to use for classification
        model_name: Name of the model to use
        
    Returns:
        str: One of "satisfied", "violated", "not_mentioned"
    """
    system_prompt = f"""You are a legal expert specializing in GDPR compliance analysis. Your task is to determine whether a Data Processing Agreement (DPA) segment satisfies, violates, or does not mention a specific GDPR regulatory requirement.

Instructions:
1. Analyze the DPA segment in relation to the given GDPR requirement
2. Classify the relationship as one of three categories:
   - "satisfied": The DPA segment explicitly addresses and fulfills the requirement
   - "violated": The DPA segment contradicts or fails to meet the requirement
   - "not_mentioned": The DPA segment does not address the requirement at all

3. Respond with ONLY one word: satisfied, violated, or not_mentioned
4. Do not provide explanations or additional text

Examples:

Example 1 (Satisfied):
REQUIREMENT: The processor shall ensure that persons authorized to process personal data have committed themselves to confidentiality or are under an appropriate statutory obligation of confidentiality.
DPA SEGMENT: The Processor shall ensure that every employee authorized to process Customer Personal Data is subject to a contractual duty of confidentiality.
CLASSIFICATION: satisfied

Example 2 (Not Mentioned):
REQUIREMENT: The processor shall encrypt personal data during transmission and at rest.
DPA SEGMENT: This DPA shall remain in effect so long as processor processes Personal Data, notwithstanding the expiration or termination of the Agreement.
CLASSIFICATION: not_mentioned

Example 3 (Violated):
REQUIREMENT: The processor shall encrypt personal data during transmission and at rest.
DPA SEGMENT: The processor stores all customer data in plain text format without any encryption.
CLASSIFICATION: violated"""

    user_prompt = f"""REQUIREMENT: {requirement_text}
DPA SEGMENT: {dpa_segment}
CLASSIFICATION:"""

    response = llm_model.generate(user_prompt, model_name=model_name, system_prompt=system_prompt)
    
    # Filter out <think> sections from reasoning models
    response = filter_think_sections(response)
    
    # Clean and validate response
    response = response.strip().lower()
    
    # Map variations to standard labels
    if response in ["satisfied", "satisfies", "fulfill", "fulfills", "meets"]:
        return "satisfied"
    elif response in ["violated", "violates", "contradiction", "contradicts", "fails"]:
        return "violated"
    elif response in ["not_mentioned", "not mentioned", "no mention", "irrelevant", "unrelated"]:
        return "not_mentioned"
    else:
        # Default to not_mentioned if response is unclear
        print(f"Warning: Unclear response '{response}', defaulting to 'not_mentioned'")
        return "not_mentioned"

def main():
    parser = argparse.ArgumentParser(description="Baseline DPA completeness evaluation using direct LLM classification")
    parser.add_argument("--model", type=str, default="llama3.3:70b",
                        help="Ollama model to use for classification")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--output", type=str, default="results/baseline_results",
                        help="Output directory for baseline results")
    parser.add_argument("--target_dpas", type=str, default="Online 124,Online 132",
                        help="Comma-separated list of target DPAs to process")
    parser.add_argument("--req_ids", type=str, default="all",
                        help="Comma-separated list of requirement IDs to process, or 'all'")
    parser.add_argument("--max_segments", type=int, default=0,
                        help="Maximum number of segments to process per DPA (0 means all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize Ollama client
    print(f"Initializing Ollama client with model: {args.model}")
    llm_model = OllamaClient()
    if not llm_model.check_health():
        print("Error: Ollama server is not running. Please start it first.")
        return
    
    # Load requirements
    print("Loading ground truth requirements...")
    requirements = load_requirements()
    print(f"Loaded {len(requirements)} requirements")
    
    # Filter requirements if specified
    if args.req_ids.lower() != "all":
        req_ids = [int(id.strip()) for id in args.req_ids.split(",")]
        requirements = {id: requirements[id] for id in req_ids if id in requirements}
        print(f"Using {len(requirements)} specified requirements: {list(requirements.keys())}")
    
    # Load DPA segments
    print(f"Loading DPA segments from: {args.dpa}")
    df = pd.read_csv(args.dpa)
    
    # Parse target DPAs
    target_dpas = [dpa.strip() for dpa in args.target_dpas.split(",")]
    
    # Process each DPA
    all_results = []
    
    for target_dpa in target_dpas:
        print(f"\nProcessing DPA: {target_dpa}")
        
        # Filter for the target DPA
        df_filtered = df[df['DPA'] == target_dpa].copy()
        
        if df_filtered.empty:
            print(f"Warning: No segments found for DPA '{target_dpa}'")
            continue
        
        # Apply segment limit if specified
        if args.max_segments > 0:
            df_filtered = df_filtered.head(args.max_segments)
        
        print(f"Processing {len(df_filtered)} segments")
        
        # Create deolingo-compatible results file
        deolingo_format_file = os.path.join(args.output, f"baseline_deolingo_results_{target_dpa.replace(' ', '_')}.txt")
        
        with open(deolingo_format_file, 'w') as deolingo_file:
            # Process each requirement
            for req_id, req_text in tqdm(requirements.items(), desc=f"Processing requirements for {target_dpa}"):
                r_label = req_number_to_r_label(req_id)
                
                # Process each segment
                for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), 
                                   desc=f"Processing segments for requirement {req_id}", leave=False):
                    segment_id = row["ID"]
                    segment_text = row["Sentence"]
                    
                    # Get ground truth if available (check for R10-R29 pattern)
                    ground_truth = None
                    for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
                        if col in row and pd.notna(row[col]):
                            mapped_req = map_requirement_labels(row[col])
                            if mapped_req == req_id:
                                ground_truth = "satisfied"
                                break
                    
                    if ground_truth is None:
                        # Check if it's marked as "other" (meaning not mentioned)
                        if any(row.get(col, "") == "other" for col in ["Requirement-1", "Requirement-2", "Requirement-3"]):
                            ground_truth = "not_mentioned"
                        else:
                            ground_truth = "not_mentioned"  # Default assumption
                    
                    if args.verbose:
                        print(f"\nSegment {segment_id} for requirement {req_id}:")
                        print(f"Text: {segment_text[:100]}...")
                        print(f"Ground truth: {ground_truth}")
                    
                    # Classify using LLM
                    predicted = classify_dpa_segment(req_text, segment_text, llm_model, args.model)
                    
                    if args.verbose:
                        print(f"Predicted: {predicted}")
                    
                    # Store result
                    result = {
                        "dpa": target_dpa,
                        "segment_id": segment_id,
                        "requirement_id": req_id,
                        "requirement_text": req_text,
                        "segment_text": segment_text,
                        "ground_truth": ground_truth,
                        "predicted": predicted,
                        "correct": predicted == ground_truth
                    }
                    
                    all_results.append(result)
                    
                    # Write to deolingo-compatible format file
                    deolingo_file.write(f"Processing DPA {target_dpa}, Requirement {req_id}, Segment {segment_id}...\n")
                    if predicted == "satisfied":
                        deolingo_file.write(f"FACTS: status({predicted})\n")
                    else:
                        deolingo_file.write(f"FACTS:\n")  # Empty facts for not_mentioned/violated
                    deolingo_file.write("--------------------------------------------------\n")
    
    # Save detailed results
    results_file = os.path.join(args.output, f"baseline_results_{args.model.replace(':', '_')}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBaseline evaluation completed!")
    print(f"Results saved to: {results_file}")
    print(f"Deolingo-compatible format files saved in: {args.output}")
    
    # Print summary statistics
    total = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nSummary:")
    print(f"Total evaluations: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Print per-class statistics
    from collections import defaultdict
    by_label = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for result in all_results:
        label = result["ground_truth"]
        by_label[label]["total"] += 1
        if result["correct"]:
            by_label[label]["correct"] += 1
    
    print("\nPer-class accuracy:")
    for label, stats in by_label.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {label}: {acc:.3f} ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    main() 