#!/usr/bin/env python3
"""
Test runner for RAGBuilder v2 testing suite.
Runs all tests and generates a summary report.
"""

import os
import sys
import unittest
import time
from datetime import datetime
import json
from pathlib import Path
import argparse
import importlib.util

def import_module(module_name, file_path):
    """Safely import a module, returning None if import fails"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec:
            print(f"Could not find module {module_name} at {file_path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        return None

def load_test_modules():
    """Load available test modules"""
    current_dir = Path.cwd()
    modules = []
    
    # Check for test_minimal.py
    minimal_path = current_dir / "test_minimal.py"
    if minimal_path.exists():
        minimal_module = import_module("test_minimal", minimal_path)
        if minimal_module:
            modules.append(minimal_module)
    
    # Check for test_ragbuilder.py
    basic_path = current_dir / "test_ragbuilder.py"
    if basic_path.exists():
        basic_module = import_module("test_ragbuilder", basic_path)
        if basic_module:
            modules.append(basic_module)
    
    # Check for test_ragbuilder_advanced.py
    adv_path = current_dir / "test_ragbuilder_advanced.py"
    if adv_path.exists():
        adv_module = import_module("test_ragbuilder_advanced", adv_path)
        if adv_module:
            modules.append(adv_module)
    
    # Check for test_ragbuilder_edge_cases.py
    edge_path = current_dir / "test_ragbuilder_edge_cases.py"
    if edge_path.exists():
        edge_module = import_module("test_ragbuilder_edge_cases", edge_path)
        if edge_module:
            modules.append(edge_module)
    
    return modules

def run_tests(test_modules, verbosity=2, output_dir=None):
    """Run the specified test modules and generate a report"""
    if not test_modules:
        print("No test modules were successfully loaded.")
        return 1
    
    # Set up output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary for storing test outcomes
    results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0
        },
        "modules": {}
    }
    
    # Run each test module and collect results
    start_time = time.time()
    for module in test_modules:
        module_name = module.__name__
        print(f"\n--- Running tests in {module_name} ---\n")
        
        # Create a test suite from the module
        suite = unittest.TestLoader().loadTestsFromModule(module)
        
        # Run the tests and collect results
        module_result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
        
        # Store results for this module
        results["modules"][module_name] = {
            "total": module_result.testsRun,
            "passed": module_result.testsRun - len(module_result.failures) - len(module_result.errors) - len(module_result.skipped),
            "failed": len(module_result.failures),
            "errors": len(module_result.errors),
            "skipped": len(module_result.skipped),
            "failures": [f"{failure[0]}: {str(failure[1])}" for failure in module_result.failures],
            "error_details": [f"{error[0]}: {str(error[1])}" for error in module_result.errors]
        }
        
        # Update summary stats
        results["summary"]["total"] += module_result.testsRun
        results["summary"]["passed"] += (module_result.testsRun - len(module_result.failures) - 
                                        len(module_result.errors) - len(module_result.skipped))
        results["summary"]["failed"] += len(module_result.failures)
        results["summary"]["errors"] += len(module_result.errors)
        results["summary"]["skipped"] += len(module_result.skipped)
    
    # Calculate total runtime
    end_time = time.time()
    results["runtime_seconds"] = round(end_time - start_time, 2)
    
    # Print summary
    print("\n--- Test Summary ---")
    print(f"Total tests: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Errors: {results['summary']['errors']}")
    print(f"Skipped: {results['summary']['skipped']}")
    print(f"Runtime: {results['runtime_seconds']} seconds")
    
    # Save results to file if output directory is specified
    if output_dir:
        # JSON report
        json_path = Path(output_dir) / "test_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Text summary
        text_path = Path(output_dir) / "test_summary.txt"
        with open(text_path, "w") as f:
            f.write("=== RAGBuilder Test Results ===\n")
            f.write(f"Run at: {results['timestamp']}\n")
            f.write(f"Runtime: {results['runtime_seconds']} seconds\n\n")
            
            f.write("--- Summary ---\n")
            f.write(f"Total tests: {results['summary']['total']}\n")
            f.write(f"Passed: {results['summary']['passed']}\n")
            f.write(f"Failed: {results['summary']['failed']}\n")
            f.write(f"Errors: {results['summary']['errors']}\n")
            f.write(f"Skipped: {results['summary']['skipped']}\n\n")
            
            f.write("--- Details by Module ---\n")
            for module_name, module_results in results["modules"].items():
                f.write(f"\n{module_name}:\n")
                f.write(f"  Total: {module_results['total']}\n")
                f.write(f"  Passed: {module_results['passed']}\n")
                f.write(f"  Failed: {module_results['failed']}\n")
                f.write(f"  Errors: {module_results['errors']}\n")
                f.write(f"  Skipped: {module_results['skipped']}\n")
                
                if module_results['failures']:
                    f.write("\n  Failures:\n")
                    for failure in module_results['failures']:
                        f.write(f"    - {failure}\n")
                
                if module_results['error_details']:
                    f.write("\n  Errors:\n")
                    for error in module_results['error_details']:
                        f.write(f"    - {error}\n")
        
        print(f"\nDetailed results saved to {output_dir}")
    
    # Return exit code (0 if all tests passed, 1 otherwise)
    return 0 if results["summary"]["failed"] == 0 and results["summary"]["errors"] == 0 else 1

def main():
    """Parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run RAGBuilder tests")
    parser.add_argument("--verbosity", type=int, default=2, help="Test output verbosity (1-3)")
    parser.add_argument("--output", type=str, default="test_results", help="Directory to store test results")
    parser.add_argument("--minimal-only", action="store_true", help="Run only minimal import tests")
    args = parser.parse_args()
    
    # Load available test modules
    available_modules = load_test_modules()
    
    # Filter modules based on command-line arguments
    test_modules = []
    
    if args.minimal_only:
        for module in available_modules:
            if module.__name__ == "test_minimal":
                test_modules.append(module)
                break
    else:
        test_modules = available_modules
    
    if not test_modules:
        print("No test modules were available to run.")
        return 1
    
    # Run the tests
    exit_code = run_tests(
        test_modules=test_modules,
        verbosity=args.verbosity,
        output_dir=args.output
    )
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 