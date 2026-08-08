import argparse
import json
import sys
import os
from src.parser import load_prompts, load_function_defs
from pydantic import ValidationError # type: ignore (ModuleNotFoundError: No module named 'pydantic')
from llm_sdk import Small_LLM_Model
import numpy as np # type: ignore (ModuleNotFoundError: No module named 'numpy')
from src.decoder import load_vocabulary, build_prompt, get_allowed_token_ids


def main():
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="LLM Constrained Decoding Engine")
    
    # 2. Add the arguments with the required defaults
    parser.add_argument(
        "--functions_definition", 
        type=str, 
        default="data/input2/functions_definition.json",
        help="Path to the functions definition JSON file"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/input2/function_calling_tests.json",
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
            # 🚨 THE STATE MACHINE 🚨
            # State 0: Forcing the start of the JSON and the "name" key
            # =================================================================
            # 🚨 THE REBUILT STATE MACHINE 🚨
# =================================================================
            # 🚨 THE REBUILT STATE MACHINE (MINIFIED JSON) 🚨
            PREFIX = '{"name":"'  # <-- Space removed!
            
            # State 0 & 1: Force the prefix, the name, AND the transition to parameters
            if '","parameters":{' not in generated_json_string: # <-- Spaces removed!
                
                # Build the FULL valid paths from the very beginning. 
                # Target example: '{"name":"fn_add_numbers","parameters":{'
                valid_targets = [f'{PREFIX}{fn.name}","parameters":{{' for fn in definitions]
                
                allowed_ids = get_allowed_token_ids(vocab, valid_targets, generated_json_string)
                
                if allowed_ids:
                    mask[allowed_ids] = logits_array[allowed_ids]
                else:
                    # A safety net so we don't infinite scream. 
                    print(f"\n[DEBUG ERROR] No allowed IDs found for string: {generated_json_string}")
                    break

            # State 2: We are inside the parameters!
            else:
                # 1. Figure out which function the LLM chose
                temp_string = generated_json_string.split('{"name":"')[1]
                chosen_function_name = temp_string.split('","parameters":{')[0]
                
                # Find the current definition
                current_def = None
                for fn in definitions:
                    if fn.name == chosen_function_name:
                        current_def = fn
                        break

                # 2. Extract the content inside the parameters dictionary so far
                params_string = generated_json_string.split('","parameters":{')[1]
                
                # Splitting with commas is tricky because values can contain commas and it can break the logic.
                current_chunk = params_string
                
                # =============================================================
                # STATE 2A: Spelling a KEY (No colon in the current chunk yet)
                # =============================================================
                if ":" not in current_chunk:
                    
                    # THE FIX FOR INFINITE QUOTES:
                    # We subtract the partially-spelled chunk to get a stable base history!
                    if current_chunk:
                        base_history = generated_json_string[:-len(current_chunk)]
                    else:
                        base_history = generated_json_string
                        
                    valid_targets = []
                    for param_name in current_def.parameters.keys():
                        # Only add this key if we haven't already spelled it!
                        if f'"{param_name}":' not in params_string:
                            valid_targets.append(f'{base_history}"{param_name}":')
                    
                    # If all keys are done, force closing bracket
                    if not valid_targets:
                        valid_targets = [f'{base_history}}}']

                    allowed_ids = get_allowed_token_ids(vocab, valid_targets, generated_json_string)
                    
                    if allowed_ids:
                        mask[allowed_ids] = logits_array[allowed_ids]
                    else:
                        print(f"\n[DEBUG ERROR] No allowed IDs for key spelling.")
                        break

                # =============================================================
                # STATE 2B: Spelling a VALUE (Colon exists!)
                # =============================================================
                else:
                    # The colon exists! The LLM knows it needs to output the value.
                    # For right now, let it run completely wild and guess the answer itself!
                    mask = logits_array



# chosen_function_name is now exactly "fn_add_numbers"
            # =================================================================
            # =================================================================


            # Replace this later with: mask[allowed_ids] = logits_array[allowed_ids]
            # mask = logits_array 
            
            # 4. Pick the highest allowed score
            best_token_id = int(np.argmax(mask))
            
            # 5. Update the sequence and our string tracker
            input_ids.append(best_token_id)
            new_text = model.decode([best_token_id])
            generated_json_string += new_text
            
            print(new_text, end="", flush=True)
            
            # 6. Break the loop if the JSON is finished!
            try:
                # The ultimate safety net: If Python can parse it, it is 100% valid JSON.
                # This ignores any trailing newlines or spaces the tokenizer throws at it.
                parsed_llm = json.loads(generated_json_string)
                break
            except json.JSONDecodeError:
                # JSON is not finished yet, keep looping!
                pass
        
        # 7. Save the result for this specific prompt
        final_results.append({
            "prompt": test.prompt,
            # We will parse generated_json_string into a real dict later
            "name": parsed_llm.get("name", None),
            "parameters": parsed_llm.get("parameters", None),
        })

    
    # 8. Save the final array to data/output/function_calling_results.json
    print("\n\n--- Generation Finished ---")
    
    # Isolate the directory path (e.g., "data/output") from the full file path
    output_dir = os.path.dirname(args.output)
    
    # Create the folders if they don't exist (exist_ok=True prevents crashes if they do)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Now it is completely safe to open and write the file!
    with open(args.output, 'w', encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")