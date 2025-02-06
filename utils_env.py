from pathlib import Path

def find_file_in_parents(filename: str, start_path: Path = None) -> Path:
    """
    在当前目录及所有上级目录中查找指定的文件，并返回完整路径
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    # **先检查当前目录**
    if (start_path / filename).exists():
        return start_path / filename

    # **再逐级向上查找**
    for parent in start_path.parents:
        if (parent / filename).exists():
            return parent / filename

    raise FileNotFoundError(f"未找到 `{filename}`，请检查目录结构！")

