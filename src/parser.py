import json
from pydantic import BaseModel, Field # type: ignore (ModuleNotFoundError: No module named 'pydantic')


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
        function_calling_data = json.load(f)
    return [Prompt(**item) for item in function_calling_data]


def load_function_defs(filename) -> list[FunctionDefinition]:
    with open(filename, "r") as f:
        functions_definition_data = json.load(f)
    return [FunctionDefinition(**item) for item in functions_definition_data]
