"""Execute os materiais sem alterar o ambiente dos seminários anteriores."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def main():
    tasks = {"experiment": ROOT / "experiments/scripts/experiment.py",
             "materials": ROOT / "presentation/build.py",
             "animations": ROOT / "manim/render.py",
             "notebook": ROOT / "notebooks/execute.py",
             "verify": ROOT / "experiments/scripts/verify_artifacts.py"}
    if len(sys.argv) < 2 or sys.argv[1] not in {*tasks, "all"}:
        raise SystemExit("Uso: python3 seminar_4/run.py all|experiment|materials|animations|notebook|verify")
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    env["XDG_CACHE_HOME"] = str(ROOT / ".cache")
    env["IPYTHONDIR"] = str(ROOT / ".cache/ipython")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    interpreter = ROOT / ".venv/bin/python"
    if not interpreter.exists():
        interpreter = REPO / ".venv/bin/python"
    if not interpreter.exists():
        # Ambiente existente tem link quebrado após atualização do VS Code.
        candidates = sorted((Path.home() / ".local/share/uv/python").glob(
            "cpython-3.13*-linux-*/bin/python3.13"))
        packages = REPO / ".venv/lib/python3.13/site-packages"
        if candidates and packages.exists():
            interpreter = candidates[-1]
            env["PYTHONPATH"] = str(packages) + os.pathsep + env.get("PYTHONPATH", "")
        else:
            interpreter = Path(sys.executable)
    selected = list(tasks) if sys.argv[1] == "all" else [sys.argv[1]]
    for name in selected:
        print(f"Executando: {name}", flush=True)
        subprocess.run([str(interpreter), str(tasks[name]), *sys.argv[2:]],
                       cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
