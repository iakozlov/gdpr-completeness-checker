# solver/deolingo_interface.py
import os
import tempfile
import subprocess
from typing import Dict

# Try the correct import path
try:
    from deolingo.deolingo import Solver as DeolingoSolver
except ImportError:
    try:
        # Alternative path
        from deolingo.solver import Solver as DeolingoSolver
    except ImportError:
        print("Warning: Couldn't import DeolingoSolver, falling back to command line approach")
        DeolingoSolver = None

class DeolingoInterface:
    """Interface to interact with the deolingo solver."""
    
    def prepare_program(self, requirements: Dict[str, str], dpa_segments: Dict[str, str]) -> str:
        """Prepare combined program for deolingo with requirements and DPA segments."""
        program = ""
        
        # Add requirements
        for req_id, req_repr in requirements.items():
            program += f"% Requirement {req_id}\n"
            program += f"{req_repr}\n\n"
        
        # Add DPA segments
        for seg_id, seg_repr in dpa_segments.items():
            program += f"% DPA Segment {seg_id}\n"
            program += f"{seg_repr}\n\n"
        
        # Add completeness check
        program += "% Completeness check\n"
        req_ids = list(requirements.keys())
        
        # Define satisfaction for each requirement
        for req_id, req_repr in requirements.items():
            head = req_repr.split(":-")[0].strip() if ":-" in req_repr else req_repr.strip()
            if head.endswith("."):
                head = head[:-1]
            program += f"satisfied_{req_id} :- {head}.\n"
        
        # Define completeness
        satisfied_conditions = ", ".join([f"satisfied_{req_id}" for req_id in req_ids])
        program += f"complete :- {satisfied_conditions}.\n"
        program += "#show complete/0.\n"
        program += "#show satisfied_/1.\n"
        
        return program
    
    def check_completeness(self, requirements: Dict[str, str], dpa_segments: Dict[str, str]) -> Dict:
        """Check DPA completeness against requirements."""
        # Prepare the program
        program = self.prepare_program(requirements, dpa_segments)
        
        # Try library approach first if available
        if DeolingoSolver is not None:
            try:
                solver = DeolingoSolver()
                
                # Call solve() without problematic arguments
                solutions = solver.solve(program)
                
                # Initialize results
                results = {
                    'complete': False,
                    'satisfied_requirements': set(),
                    'unsatisfied_requirements': set(),
                    'raw_output': str(solutions)
                }
                
                # Process solutions
                for solution in solutions:
                    solution_str = str(solution)
                    
                    # Check for completeness
                    if "complete" in solution_str:
                        results['complete'] = True
                    
                    # Check for satisfied requirements
                    for req_id in requirements:
                        if f"satisfied_{req_id}" in solution_str:
                            results['satisfied_requirements'].add(req_id)
                
                # Determine unsatisfied requirements
                for req_id in requirements:
                    if req_id not in results['satisfied_requirements']:
                        results['unsatisfied_requirements'].add(req_id)
                
                return results
            
            except Exception as e:
                print(f"Library approach failed: {e}, falling back to command line")
                # Fall through to command-line approach
        
        # Command-line approach as fallback
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as temp_file:
                temp_filepath = temp_file.name
                temp_file.write(program)
            
            try:
                # Run deolingo as a command-line tool
                result = subprocess.run(
                    ["deolingo", temp_filepath],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                output_str = result.stdout
                
                # Initialize results
                results = {
                    'complete': False,
                    'satisfied_requirements': set(),
                    'unsatisfied_requirements': set(),
                    'raw_output': output_str
                }
                
                # Check for completeness and satisfied requirements
                if "complete" in output_str:
                    results['complete'] = True
                
                # Check for satisfied requirements
                for req_id in requirements:
                    if f"satisfied_{req_id}" in output_str:
                        results['satisfied_requirements'].add(req_id)
                    else:
                        results['unsatisfied_requirements'].add(req_id)
                
                return results
                
            finally:
                # Clean up
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    
        except Exception as e:
            print(f"Command-line approach failed: {e}")
            return {
                'complete': False, 
                'error': str(e),
                'satisfied_requirements': set(),
                'unsatisfied_requirements': set(requirements.keys())
            }