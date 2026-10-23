"""Keep the Colab and Kaggle inference-API notebooks consistent with the code and safe to share."""

import ast
import json
import re
from pathlib import Path

import pytest

from src.data.prepare_dataset import PROJECT_ROOT

NOTEBOOKS = PROJECT_ROOT / "notebooks"
# Notebook -> how it reads the API token from the platform's secret store.
PLATFORMS = {
    "colab_inference_api.ipynb": 'userdata.get("CFI_API_TOKEN")',
    "kaggle_inference_api.ipynb": 'user_secrets.get_secret("CFI_API_TOKEN")',
}


@pytest.fixture(scope="module", params=sorted(PLATFORMS))
def notebook(request):
    return request.param


@pytest.fixture(scope="module")
def cells(notebook):
    notebook = json.loads((NOTEBOOKS / notebook).read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def sources(cells):
    return ["".join(cell["source"]) for cell in cells]


def test_committed_without_outputs(cells):
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in cells)


def test_code_cells_are_valid_python(cells):
    for source in sources(cells):
        ast.parse("\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))))


def test_api_token_comes_from_platform_secrets_and_is_never_printed(notebook, cells):
    text = "\n".join(sources(cells))
    assert PLATFORMS[notebook] in text
    assert not re.search(r"print\([^)]*api_token", text)
    assert "{api_token}" not in text.replace('f"Bearer {api_token}"', "")
    assert not re.search(r"ghp_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}", (NOTEBOOKS / notebook).read_text(encoding="utf-8"))


def test_server_binds_to_localhost_on_the_gpu(cells):
    start = next(s for s in sources(cells) if "src.serving.api" in s)
    assert '"--host", "127.0.0.1"' in start
    assert '"CFI_DEVICE": "cuda"' in start
    assert '"CFI_ADAPTER_PATH": ADAPTER_DIR' in start


def test_tunnel_is_checked_to_reject_requests_without_the_token(cells):
    check = next(s for s in sources(cells) if "PUBLIC_URL}/predict" in s)
    assert "anonymous.status_code == 401" in check


def test_commands_use_existing_modules_and_flags(cells):
    text = "\n".join(sources(cells))
    commands = re.findall(r"(?:python -m |\"-m\", \")(src(?:\.\w+)+)", text)
    assert set(commands) == {"src.serving.api", "src.serving.benchmark"}
    for flag in re.findall(r"(?<![\w-])--(?:requests|validation|reference|host|port)\b", text):
        module_sources = [(PROJECT_ROOT / Path(*m.split("."))).with_suffix(".py").read_text(encoding="utf-8")
                          for m in commands]
        assert any(f'"{flag}"' in source for source in module_sources), flag


def test_only_the_validation_split_is_used(cells):
    text = "\n".join(sources(cells))
    assert "--split test" not in text and "--final-test" not in text and "test.jsonl" not in text
    assert "checkpoint-195_validation.jsonl" in text


def test_kaggle_notebook_uses_one_gpu_and_kaggle_paths():
    text = "\n".join(sources(json.loads((NOTEBOOKS / "kaggle_inference_api.ipynb").read_text(encoding="utf-8"))["cells"]))
    assert '"CUDA_VISIBLE_DEVICES": "0"' in text
    assert "/kaggle/input" in text and "google.colab" not in text
