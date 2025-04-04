import os

from llava.utils import io

__all__ = ["EVAL_ROOT", "TASKS"]


EVAL_ROOT = "scripts/eval"
print(os.path.dirname(__file__))
TASKS = io.load(os.path.join(os.path.dirname(__file__), "registry.yaml"))


