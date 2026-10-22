"""Keep the Colab notebook consistent with the code and the evaluation rules."""

import ast
import json
import re
from pathlib import Path

import pytest

from src.data.prepare_dataset import PROJECT_ROOT

NOTEBOOK = PROJECT_ROOT / "notebooks" / "colab_train_eval.ipynb"
COMMAND = re.compile(r"python -m (src(?:\.\w+)+)([^\"\n]*)")


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def joined_strings(source):
    """Join implicitly concatenated string literals split across lines."""
    return re.sub(r'"\s*\n\s*f?"', "", source)


def settings(notebook):
    namespace = {}
    exec(code_cells(notebook)[0], namespace)
    return namespace


def test_notebook_is_committed_without_outputs(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_code_cells_are_valid_python(notebook):
    for source in code_cells(notebook):
        lines = [line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))]
        ast.parse("\n".join(lines))


def test_expensive_and_final_steps_are_off_by_default(notebook):
    namespace = settings(notebook)
    assert namespace["RUN_TRAINING"] is False
    assert namespace["RUN_FINAL_TEST"] is False


def test_paths_match_training_config(notebook):
    import yaml

    config = yaml.safe_load((PROJECT_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    assert settings(notebook)["RUN_DIR"] == config["training"]["output_dir"]


def test_every_command_uses_existing_modules_and_flags(notebook):
    commands = [match for source in code_cells(notebook) for match in COMMAND.findall(joined_strings(source))]
    assert {module for module, _ in commands} >= {
        "src.data.build_sft_dataset",
        "src.modeling.train",
        "src.modeling.inference",
        "src.evaluation.metrics",
        "src.evaluation.select_checkpoint",
    }
    for module, arguments in commands:
        source = (PROJECT_ROOT / Path(*module.split("."))).with_suffix(".py").read_text(encoding="utf-8")
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", arguments):
            assert f'"{flag}"' in source, f"{module} has no {flag} option"


def test_test_split_is_only_used_in_the_gated_final_cell(notebook):
    test_cells = [s for s in code_cells(notebook) if "--split test" in s or "--final-test" in s]
    assert len(test_cells) == 1
    final = test_cells[0]
    assert "if RUN_FINAL_TEST:" in final
    assert "checkpoint_selection.json" in final
    # Test predictions are generated only if they do not exist yet.
    assert "if not os.path.exists(predictions):" in final


def test_checkpoint_selection_runs_on_validation_without_final_test(notebook):
    selection = [s for s in code_cells(notebook) if "src.evaluation.select_checkpoint" in s]
    assert selection and all("--final-test" not in s and "--split" not in s for s in selection)


def test_training_refuses_to_overwrite_existing_checkpoints(notebook):
    cell = next(s for s in code_cells(notebook) if 'run("python -m src.modeling.train")' in s)
    assert "if RUN_TRAINING:" in cell
    assert cell.index("already has checkpoints") < cell.index('run("python -m src.modeling.train")')


def test_github_token_comes_from_colab_secrets_and_is_not_stored(notebook):
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"ghp_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}|gho_[A-Za-z0-9]{10,}", text)
    clone = next(s for s in code_cells(notebook) if "userdata.get(\"GITHUB_TOKEN\")" in s)
    assert 'replace(token, "***")' in clone
    assert 'git("remote", "set-url", "origin", f"https://github.com/{REPO}.git")' in clone


def test_run_helper_stops_on_failure(notebook):
    class Shell:
        def __init__(self, exit_code):
            self.user_ns, self.exit_code, self.commands = {}, exit_code, []

        def system(self, command):
            self.commands.append(command)
            self.user_ns["_exit_code"] = self.exit_code

    namespace = {}
    for exit_code, should_raise in ((0, False), (1, True)):
        shell = Shell(exit_code)
        namespace["get_ipython"] = lambda shell=shell: shell
        exec(code_cells(notebook)[0], namespace)
        if should_raise:
            with pytest.raises(RuntimeError, match="command failed"):
                namespace["run"]("python -m src.modeling.train")
        else:
            namespace["run"]("python -m src.modeling.train")
        assert shell.commands == ["python -m src.modeling.train"]
