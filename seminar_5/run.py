"""Run the seminar stages in an isolated, explicitly selected Python environment."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    """Run one stage or the complete workflow.

    Raises:
        SystemExit: If the command or Python environment is unavailable.
    """
    tasks = {
        "experiment": "experiments/scripts/experiment.py",
        "figures": "presentation/figures.py",
        "animations": "manim/render.py",
        "materials": "presentation/build.py",
        "notebooks": "notebooks/build.py",
        "verify": "experiments/scripts/verify.py",
    }
    if len(sys.argv) != 2 or sys.argv[1] not in {*tasks, "all"}:
        raise SystemExit("Uso: python3 seminar_5/run.py all|" + "|".join(tasks))
    env = os.environ.copy()
    local = ROOT / ".venv/bin/python"
    repo = ROOT.parent / ".venv/bin/python"
    if local.exists():
        python = local
    elif repo.exists():
        python = repo
    else:
        candidates = sorted(
            (Path.home() / ".local/share/uv/python").glob("cpython-3.13*-linux-*/bin/python3.13")
        )
        packages = ROOT.parent / ".venv/lib/python3.13/site-packages"
        if candidates and packages.exists():
            python = candidates[-1]
            env["PYTHONPATH"] = str(packages)
            print("Ambiente legado: Python uv + site-packages existente, sem modificá-lo.", flush=True)
        else:
            python = Path(sys.executable)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", ""), str(ROOT / ".deps")]
    )
    env["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    env["XDG_CACHE_HOME"] = str(ROOT / ".cache")
    env["IPYTHONDIR"] = str(ROOT / ".cache/ipython")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    names = list(tasks) if sys.argv[1] == "all" else [sys.argv[1]]
    for name in names:
        subprocess.run([str(python), str(ROOT / tasks[name])], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
