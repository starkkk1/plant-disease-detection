"""Start the classification API from any working directory.

Use the Python environment containing backend/requirements.txt.
Additional arguments are passed to Uvicorn, for example --port 8000.
"""
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    try:
        from uvicorn import main as uvicorn_main
    except ModuleNotFoundError as exc:
        if exc.name != "uvicorn":
            raise
        sys.exit(
            f"Uvicorn is missing in {sys.executable}. Use the ML virtual environment "
            f"or install: python -m pip install -r {project_root / 'backend/requirements.txt'}"
        )
    uvicorn_main(
        args=["backend.main:app", "--app-dir", str(project_root), *sys.argv[1:]],
        prog_name="run_api.py",
    )


if __name__ == "__main__":
    main()
