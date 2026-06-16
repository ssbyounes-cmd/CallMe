import argparse
import json
import sys
from src.parser import load_prompts, load_function_defs
from pydantic import ValidationError # type: ignore (ModuleNotFoundError: No module named 'pydantic')
from llm_sdk import Small_LLM_Model


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



    print("Loading the Qwen3-0.6B model...")
    model = Small_LLM_Model()
    system_prompt = """What is the sum of 4 and 5?"""

    # 1. Convert the text prompt into a Tensor of Token IDs, then to a standard Python list
    input_ids_tensor = model.encode(system_prompt)
    input_ids = input_ids_tensor[0].tolist()

    print(f"\nPrompt encoded into {len(input_ids)} tokens.")
    print("Starting generation (Greedy Decoding / NO constraints)...\n")
    
    # Print the prompt so we can see the continuation seamlessly
    print(system_prompt, end="")

    # 2. The Generation Loop (let's force it to generate exactly 30 tokens)
    for _ in range(30):
        # Pass the current sequence to the neural network to get the 100,000+ raw scores
        logits = model.get_logits_from_input_ids(input_ids)
        
        # Pick the token ID with the absolute highest score (argmax)
        best_token_id = logits.index(max(logits))
        
        # Add the chosen token ID to our sequence so it's included in the next loop
        input_ids.append(best_token_id)
        
        # Decode ONLY the new token and print it to the terminal live
        new_text = model.decode([best_token_id])
        print(new_text, end="", flush=True)

    print("\n\n--- Generation Finished ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error occurred: {e}")