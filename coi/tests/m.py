from pathlib import Path
def find_project_root(start_path=None):
    if start_path is None:
        start_path = Path.cwd()
    
    current = Path(start_path)
    for _ in range(10):
        if (current / 'setup.py').exists() and (current / 'tests').exists():
            return str(current)
        if current.parent == current:
            break
        current = current.parent
    return str(Path.cwd())