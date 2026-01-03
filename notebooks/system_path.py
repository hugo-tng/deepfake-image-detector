import sys
from pathlib import Path

def setup():
    project_root = Path(Path(__file__).resolve().parent).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"Project root set to: {project_root}")