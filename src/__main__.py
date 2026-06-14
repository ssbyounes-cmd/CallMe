import argparse
import json
import sys
from src.parser import load_prompts, load_function_defs
from pydantic import ValidationError # type: ignore (ModuleNotFoundError: No module named 'pydantic')

def main():
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="LLM Constrained Decoding Engine")
    
    # 2. Add the arguments with the required defaults
    parser.add_argument(
        "--functions_definition", 
        type=str, 
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/input/function_calling_tests.json",
        help="Path to the prompts test JSON file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/output/function_calling_results.json",
        help="Path to where the results should be saved"
    )

    # 3. Parse the commands typed in the terminal
    args = parser.parse_args()

    # 4. Pass the paths (whether default or custom) to your parsing functions
    print(f"Loading definitions from: {args.functions_definition}")
    try:
        definitions = load_function_defs(args.functions_definition)
        
        print(f"Loading tests from: {args.input}")
        tests = load_prompts(args.input)
        print("Files parsed successfully!")
    except OSError as e:
        print(f"Error opening file {e.filename}: {e.strerror}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Validation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error occurred: {e}")