from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "analytical_gscr_walkthrough.ipynb"
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter_local" / "ipython"))
os.environ.setdefault(
    "JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter_local" / "runtime")
)


def main() -> None:
    Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    with NOTEBOOK.open("r", encoding="utf-8") as stream:
        notebook = nbformat.read(stream, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    with NOTEBOOK.open("w", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)
    print(f"Executed: {NOTEBOOK}")


if __name__ == "__main__":
    main()
