import time
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any


@contextmanager
def timer(label: str = ""):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if label:
        print(f"[{label}] {elapsed:.4f}s")


def measure(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


class CSVLogger:
    def __init__(self, path: Path, fieldnames: list):
        self.path = path
        self.fieldnames = fieldnames
        self._file = None
        self._writer = None

    def __enter__(self):
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, *_):
        if self._file:
            self._file.close()

    def write(self, row: Dict[str, Any]):
        self._writer.writerow(row)
        self._file.flush()


def save_json(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
