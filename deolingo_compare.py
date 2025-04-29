# deolingo_compare.py
import os
import subprocess
import tempfile
from enum import Enum, auto

class ComplianceResult(Enum):
    """Represents the outcome of the compliance check."""
    SATISFIED = auto()
    VIOLATED = auto()
    ERROR = auto()
    UNKNOWN = auto()

def check_dpa_compliance(
    requirement_asp: str,
    dpa_segment_asp: str,
    requirement_id: str
):
    """
    Compares a DPA segment against a regulatory requirement using Deolingo command-line.
    
    Args:
        requirement_asp: The symbolic representation of the requirement
        dpa_segment_asp: The symbolic representation of the DPA segment
        requirement_id: Identifier for the requirement
        
    Returns:
        A tuple containing:
        - ComplianceResult enum (SATISFIED, VIOLATED, ERROR, UNKNOWN)
        - The raw output from deolingo
    """
    try:
        # Create a program with both requirement and DPA segment
        program = f"""
% --- Regulatory Requirement {requirement_id} ---
{requirement_asp}

% --- DPA Segment ---
{dpa_segment_asp}

% --- Violation Check ---
violation :- &forbidden{{X}}, X.
compliant :- not violation.

% --- Show Directives ---
#show violation/0.
#show compliant/0.
"""
        
        # Create a temporary file to save the program
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as temp_file:
            temp_filepath = temp_file.name
            temp_file.write(program)
            print(f"Saved program to temporary file: {temp_filepath}")
        
        try:
            # Run deolingo as a command-line process
            # First try to find the deolingo executable in the sibling directory
            deolingo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deolingo", "deolingo")
            
            # If not found, assume deolingo is in PATH
            if not os.path.exists(deolingo_path):
                deolingo_path = "deolingo"
                
            print(f"Using deolingo at: {deolingo_path}")
            
            # Run the command
            cmd = [deolingo_path, temp_filepath]
            print(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Process the output
            stdout = result.stdout
            stderr = result.stderr
            
            if result.returncode != 0 and stderr:
                print(f"Deolingo returned error (code {result.returncode}):")
                print(stderr)
                return ComplianceResult.ERROR, f"Deolingo error: {stderr}"
            
            # Determine the result based on the output
            if "violation" in stdout:
                return ComplianceResult.VIOLATED, stdout
            elif "compliant" in stdout or "SATISFIABLE" in stdout:
                return ComplianceResult.SATISFIED, stdout
            else:
                return ComplianceResult.UNKNOWN, stdout
                
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                
    except Exception as e:
        return ComplianceResult.ERROR, f"Error: {str(e)}"