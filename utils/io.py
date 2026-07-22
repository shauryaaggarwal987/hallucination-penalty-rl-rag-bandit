import json
from pathlib import Path
from typing import List, Dict, Any, Union

def read_text_file(path: Union[str, Path]) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def list_documents(data_dir: Union[str, Path]) -> List[Path]:
    p = Path(data_dir)
    # Recursive search for .txt and .pdf
    files = list(p.rglob("*.txt")) + list(p.rglob("*.pdf"))
    return files

def safe_write_json(path: Union[str, Path], obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def safe_read_json(path: Union[str, Path], default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default

def write_jsonl(path: Union[str, Path], iterable_of_dicts: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as writer:
        for row in iterable_of_dicts:
            writer.write(json.dumps(row, ensure_ascii=False) + '\n')

def read_jsonl(path: Union[str, Path]) -> List[Dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding='utf-8') as reader:
        return [json.loads(line) for line in reader if line.strip()]
