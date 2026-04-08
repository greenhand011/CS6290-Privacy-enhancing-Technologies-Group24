from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_dir() -> Path:
    return PROJECT_ROOT / "aclImdb"


def get_processing_output_dir() -> Path:
    return PROJECT_ROOT / "result" / "dataProcessing"


def get_data_understanding_output_dir() -> Path:
    return PROJECT_ROOT / "result" / "dataUnderstanding"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
