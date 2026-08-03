import argparse
import json
import sys
from src.parser import load_prompts, load_function_defs
from pydantic import ValidationError # type: ignore (ModuleNotFoundError: No module named 'pydantic')
from llm_sdk import Small_LLM_Model
import numpy as np # type: ignore (ModuleNotFoundError: No module named 'numpy')
from src.decoder import load_vocabulary, build_prompt


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
    vocab = load_vocabulary(model) # Do this ONCE outside the loop

    final_results = [] # Python will store the final outputs here

    # 1. Loop through the tests one by one
    for test in tests:
        print(f"\nProcessing prompt: {test.prompt}")
        
        # 2. Build the context for THIS specific test
        system_prompt = build_prompt(definitions, test.prompt)
        input_ids = model.encode(system_prompt)[0].tolist()
        
        generated_json_string = "" # This tracks what the LLM is spelling!
        
        # 3. The Constrained Generation Loop
        for _ in range(100): # Max tokens to prevent infinite loops
            logits = model.get_logits_from_input_ids(input_ids)
            logits_array = np.array(logits)
            
            # Create a mask of negative infinity
            mask = np.full(logits_array.shape, float('-inf'))
            
            # =================================================================
            # 🚨 THE STATE MACHINE GOES HERE 🚨
            # Look at `generated_json_string`. What state are we in?
            # 
            # Example Logic:
            # if generated_json_string == "":
            #     target_string = '{"name": "'
            #     allowed_ids = get_allowed_token_ids(vocab, target_string, generated_json_string)
            # elif ...
            # =================================================================
            
            # For now, to stop the code from crashing while we build it, 
            # let's just pretend ALL tokens are allowed (Unconstrained)
            # Replace this later with: mask[allowed_ids] = logits_array[allowed_ids]
            mask = logits_array 
            
            # 4. Pick the highest allowed score
            best_token_id = int(np.argmax(mask))
            
            # 5. Update the sequence and our string tracker
            input_ids.append(best_token_id)
            new_text = model.decode([best_token_id])
            generated_json_string += new_text
            
            print(new_text, end="", flush=True)
            
            # 6. Break the loop if the JSON is finished!
            if generated_json_string.endswith("}"):
                break
        
        # 7. Save the result for this specific prompt
        final_results.append({
            "prompt": test.prompt,
            # We will parse generated_json_string into a real dict later
            "result_string": generated_json_string 
        })
    
    # 8. Save the final array to data/output/function_calling_results.json
    print("\n\n--- Generation Finished ---")
    # with open(args.output, 'w') as f:
    #     json.dump(final_results, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")