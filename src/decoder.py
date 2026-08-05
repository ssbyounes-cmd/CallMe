import json
from llm_sdk import Small_LLM_Model

def load_vocabulary(model: Small_LLM_Model) -> dict[str, int]:
    """Loads the model's vocabulary file into a dictionary mapping strings to IDs."""
    vocab_path = model.get_path_to_vocab_file()
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        # This returns a dictionary like {"Hello": 1234, "fn_": 567, ...}
        return json.load(f)


def build_prompt(definitions, user_prompt):
    """Builds the context menu for a SINGLE test."""
    context = "You have access to the following functions:\n"
    
    for fn in definitions:
        context += f"- Function: {fn.name}\n"
        context += f"  Description: {fn.description}\n"
        
        # If the function has parameters, list them out cleanly
        if fn.parameters:
            context += "  Parameters:\n"
            for param_name, param_info in fn.parameters.items():
                context += f"    - {param_name} ({param_info.type})\n"
        else:
            context += "  Parameters: None\n"
            
    context += f"\nUser question: {user_prompt}\n"
    context += "Response:\n"
    
    return context


def get_allowed_token_ids(
    vocab: dict[str, int], 
    valid_targets: list[str], 
    current_string: str
) -> list[int]:
    allowed_ids = []
    for token_str, token_id in vocab.items():
        for target in valid_targets:
            if target.startswith(current_string):
                remaining_target = target[len(current_string):]
                
                # Case 1: The token fits completely inside the remaining target
                if remaining_target.startswith(token_str):
                    allowed_ids.append(token_id)
                    break
                    
                # Case 2: The token is LARGER than the remaining target, 
                # but it perfectly bridges the gap! (You missed this part)
                elif token_str.startswith(remaining_target):
                    allowed_ids.append(token_id)
                    break
                    
    return allowed_ids