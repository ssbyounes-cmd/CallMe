import json
from llm_sdk import Small_LLM_Model

def load_vocabulary(model: Small_LLM_Model) -> dict[str, int]:
    """Loads the model's vocabulary file into a dictionary mapping strings to IDs."""
    vocab_path = model.get_path_to_vocab_file()
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        # This returns a dictionary like {"Hello": 1234, "fn_": 567, ...}
        return json.load(f)