import json
from pydantic import BaseModel, Field # type: ignore (ModuleNotFoundError: No module named 'pydantic')


def dict_raise_on_duplicates(ordered_pairs):
    d = {}
    for key, value in ordered_pairs:
        if key in d:
            raise ValueError(f"Duplicate key found: {key}")
        d[key] = value
    return d


class Prompt(BaseModel):
    prompt: str


class FunctionReturns(BaseModel):
    type: str


class ParametersType(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParametersType]
    returns: FunctionReturns


def load_prompts(filename) -> list[Prompt]:
    with open(filename, "r") as f:
        function_calling_data = json.load(f, object_pairs_hook=dict_raise_on_duplicates)
    return [Prompt(**item) for item in function_calling_data]


def load_function_defs(filename) -> list[FunctionDefinition]:
    with open(filename, "r") as f:
        functions_definition_data = json.load(f, object_pairs_hook=dict_raise_on_duplicates)
    return [FunctionDefinition(**item) for item in functions_definition_data]
