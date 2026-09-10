"""Complete local-device runs over operations and packed matrix workloads."""

from .runner import run_suite
from .specification import cells, specification, specification_hash

__all__ = ["cells", "run_suite", "specification", "specification_hash"]
