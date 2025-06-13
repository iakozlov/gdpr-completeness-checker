#!/usr/bin/env python3
"""
test_rcv_integration.py

Test script to verify the RCV integration works correctly with the shell script approach.
This tests the complete pipeline similar to the existing approach.
"""

import os
import subprocess
import sys

def main():
    """Test the RCV integration with the shell script approach."""
    
    print("========== Testing RCV Integration ==========")
    
    # Check if requirements file exists
    requirements_file = "data/requirements/requirements_deontic_ai_generated.json"
    if not os.path.exists(requirements_file):
        print(f"Error: Requirements file not found: {requirements_file}")
        return 1
    
    # Check if DPA segments file exists
    dpa_segments_file = "data/test_set.csv"
    if not os.path.exists(dpa_segments_file):
        print(f"Error: DPA segments file not found: {dpa_segments_file}")
        return 1
    
    # Test parameters
    target_dpa = "Online 124"
    max_segments = 3  # Test with just 3 segments for speed
    output_dir = "results/rcv_integration_test"
    
    print(f"Testing with {max_segments} segments from DPA: {target_dpa}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    
    # Test 1: Generate RCV LP files
    print("\n[Test 1] Testing LP file generation...")
    cmd1 = [
        "python3", "generate_rcv_lp_files.py",
        "--requirements", requirements_file,
        "--dpa_segments", dpa_segments_file,
        "--target_dpa", target_dpa,
        "--output", f"{output_dir}/lp_files_test",
        "--model", "llama3.3:70b",
        "--max_segments", str(max_segments),
        "--verbose"
    ]
    
    print(f"Running: {' '.join(cmd1)}")
    try:
        result1 = subprocess.run(cmd1, check=False, capture_output=True, text=True)
        if result1.returncode == 0:
            print("✓ LP file generation test passed")
            
            # Check if LP files were created
            lp_dir = f"{output_dir}/lp_files_test"
            if os.path.exists(lp_dir):
                lp_files = [f for f in os.listdir(lp_dir) if f.endswith('.lp')]
                print(f"  Generated {len(lp_files)} LP files")
                
                # Show a sample LP file
                if lp_files:
                    sample_file = os.path.join(lp_dir, lp_files[0])
                    print(f"\n  Sample LP file ({lp_files[0]}):")
                    with open(sample_file, 'r') as f:
                        content = f.read()
                        print("  " + "\n  ".join(content.split('\n')[:10]) + "\n  ...")
            else:
                print("✗ LP files directory not created")
                return 1
        else:
            print(f"✗ LP file generation failed: {result1.stderr}")
            return 1
            
    except Exception as e:
        print(f"✗ Error running LP generation test: {e}")
        return 1
    
    # Test 2: Test shell script help
    print("\n[Test 2] Testing shell script...")
    try:
        # Just test that the shell script can be called
        result2 = subprocess.run(["bash", "run_dpa_completeness_rcv.sh"], 
                                input="Q\n", text=True, capture_output=True)
        if "RCV Approach" in result2.stdout:
            print("✓ Shell script test passed")
        else:
            print(f"✗ Shell script test failed")
            print(f"Output: {result2.stdout}")
            print(f"Error: {result2.stderr}")
            return 1
            
    except Exception as e:
        print(f"✗ Error running shell script test: {e}")
        return 1
    
    # Test 3: Check existing evaluation scripts exist
    print("\n[Test 3] Checking evaluation script dependencies...")
    evaluation_scripts = [
        "evaluate_completeness.py",
        "aggregate_evaluations.py", 
        "paragraph_metrics.py",
        "aggregate_paragraph_metrics.py"
    ]
    
    all_exist = True
    for script in evaluation_scripts:
        if os.path.exists(script):
            print(f"  ✓ {script} exists")
        else:
            print(f"  ✗ {script} missing")
            all_exist = False
    
    if not all_exist:
        print("Warning: Some evaluation scripts are missing. Full pipeline may not work.")
    
    print("\n" + "=" * 50)
    print("Integration Test Summary:")
    print("✓ RCV LP file generation works")
    print("✓ Shell script structure is correct")
    if all_exist:
        print("✓ All evaluation dependencies exist")
    else:
        print("⚠ Some evaluation dependencies missing")
    
    print("\nTo run the full RCV pipeline:")
    print(f"  ./run_dpa_completeness_rcv.sh --max_segments {max_segments}")
    print("\nNote: Make sure Ollama server is running and deolingo is installed")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 