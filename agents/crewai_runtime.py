import os
from pathlib import Path


def prepare_crewai_runtime() -> None:
    storage_root = Path.cwd() / ".crewai"
    storage_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CREWAI_STORAGE_DIR", "workspace")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("CREWAI_TESTING", "true")

    if os.name == "nt":
        os.environ["LOCALAPPDATA"] = str(storage_root)
        try:
            import appdirs

            appdirs._get_win_folder = lambda _csidl_name: str(storage_root)
        except Exception:
            pass


def get_crewai_components():
    prepare_crewai_runtime()
    try:
        from crewai import Agent, Crew, LLM, Process, Task
    except ModuleNotFoundError as exc:
        if exc.name == "crewai":
            raise ModuleNotFoundError(
                "The `crewai` package is not installed in the Python environment "
                "running this project. Install it in your active env with:\n"
                "`python -m pip install crewai langchain-openai tavily-python`"
            ) from exc
        raise

    return Agent, Crew, LLM, Process, Task
