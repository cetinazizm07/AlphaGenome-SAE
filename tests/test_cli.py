"""The stage router should only expose existing, module-level entry points."""

import ast
from pathlib import Path

from ag_sae.cli import STAGES


def test_stage_targets_exist_and_are_callable():
    for module_name, function_name in STAGES.values():
        module_path = Path(*module_name.split(".")).with_suffix(".py")
        source = Path(__file__).parents[1] / "ag_sae" / module_path
        tree = ast.parse(source.read_text())
        functions = {node.name for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert function_name in functions, f"{module_name}.{function_name} is missing"


def test_extract_and_sae_stages_use_their_real_module_entry_points():
    assert STAGES["extract"] == ("extract", "main")
    assert STAGES["sae"] == ("train", "main")
