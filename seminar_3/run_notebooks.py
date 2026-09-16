"""Execute notebooks do seminário em ordem e salve as saídas."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", help="Nomes ou números; vazio executa todos.")
    args = parser.parse_args()
    seminar = Path(__file__).resolve().parent
    cache = seminar / "data/outputs/.cache"
    for variable, child in [
        ("MPLCONFIGDIR", "matplotlib"),
        ("IPYTHONDIR", "ipython"),
        ("JUPYTER_RUNTIME_DIR", "jupyter"),
    ]:
        path = cache / child
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(path))
    paths = sorted((seminar / "notebooks").glob("*.ipynb"))
    if args.notebooks:
        paths = [
            path for path in paths
            if any(item == path.name or path.name.startswith(item) for item in args.notebooks)
        ]
    if not paths:
        raise SystemExit("Nenhum notebook selecionado.")
    for path in paths:
        print(f"Executando {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=7200,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        nbformat.write(notebook, path)
        print(f"OK: {path.name}", flush=True)
