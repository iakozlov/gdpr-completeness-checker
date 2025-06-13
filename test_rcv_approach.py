#!/usr/bin/env python3
"""
test_rcv_approach.py

Test script to demonstrate the RCV (Requirement Classification and Verification) approach
with a small subset of data.
"""

import os
import subprocess
import sys

def main():
    """Test the RCV approach with sample data."""
    
    print("========== Testing RCV Approach ==========")
    
    # Parameters for the test
    requirements_file = "data/requirements/requirements_deontic_ai_generated.json"
    dpa_segments_file = "data/test_set.csv"
    target_dpa = "Online 124"
    output_dir = "results/rcv_test"
    model = "llama3.3:70b"
    max_segments = 5  # Test with just 5 segments
    
    # Check if requirements file exists
    if not os.path.exists(requirements_file):
        print(f"Error: Requirements file not found: {requirements_file}")
        return
    
    # Check if DPA segments file exists
    if not os.path.exists(dpa_segments_file):
        print(f"Error: DPA segments file not found: {dpa_segments_file}")
        return
    
    # Construct the command
    cmd = [
        "python3", "classify_and_verify.py",
        "--requirements", requirements_file,
        "--dpa_segments", dpa_segments_file,
        "--target_dpa", target_dpa,
        "--output_dir", output_dir,
        "--model", model,
        "--max_segments", str(max_segments),
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Testing with {max_segments} segments from DPA: {target_dpa}")
    print("=" * 50)
    
    try:
        # Run the classify_and_verify script
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        
        # Check if output files were created
        expected_csv = os.path.join(output_dir, f"rcv_results_{target_dpa.replace(' ', '_')}.csv")
        expected_lp_dir = os.path.join(output_dir, "lp_files", target_dpa.replace(' ', '_'))
        
        if os.path.exists(expected_csv):
            print(f"✓ Results CSV created: {expected_csv}")
        else:
            print(f"✗ Results CSV not found: {expected_csv}")
        
        if os.path.exists(expected_lp_dir):
            lp_files = [f for f in os.listdir(expected_lp_dir) if f.endswith('.lp')]
            print(f"✓ LP files directory created: {expected_lp_dir}")
            print(f"  Generated {len(lp_files)} LP files")
        else:
            print(f"✗ LP files directory not found: {expected_lp_dir}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running the RCV script: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 