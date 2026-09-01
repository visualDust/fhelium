"""Transform arbitrary IR with selected-device inventory and build executables.

The JIT accepts mixed-level Programs. Transformation may stop at any IR state;
only an executable build asks a selected backend to cover the remaining
operations and live bindings. Mathematical CKKS correctness remains available
through separate analyses rather than being a construction prerequisite.

The default composite provider set instantiates one current Eager-backed JIT
adapter design for CPU and CUDA, with lifecycle-neutral direct logical RNS/NTT
compilation, and adds a neutral Triton provider. The Eager-backed adapter is
not a generic continuation of Compile.
"""

from ._analysis import (
    ProgramRequirements,
    analyze_requirements,
)
from ._contracts import (
    Backend,
    BackendRegistry,
    BuildResult,
    CoverageDiagnostic,
    CoverageReport,
    Executable,
    ProviderDecision,
    ProviderPolicy,
)
from ._errors import JitError, JitInputError, JitInterpreterError
from ._execution import ExecutionInputs
from ._bindings import (
    BindingResolver,
    OperationHandler,
    RuntimeBindings,
)
from ._session import Session
from .backends._interpreter_runtime import (
    InterpreterCoverage,
    InterpreterDiagnostic,
    InterpreterNotCoveredError,
    check_interpreter_coverage,
)
from .backends._interpreter import (
    InterpreterBackend,
    InterpreterExecutable,
)
from .backends._triton import (
    TritonPointwiseBackend,
    TritonPointwiseExecutable,
)
from .backends._composite import CompositeBackend, CompositeExecutable
from .planning._plan import (
    ExecutionPlan,
    OperationSupport,
    PlanRegion,
    PlanValue,
)

__all__ = [
    "Backend",
    "BackendRegistry",
    "BindingResolver",
    "BuildResult",
    "CoverageDiagnostic",
    "CoverageReport",
    "CompositeBackend",
    "CompositeExecutable",
    "Executable",
    "InterpreterBackend",
    "InterpreterCoverage",
    "InterpreterDiagnostic",
    "InterpreterExecutable",
    "InterpreterNotCoveredError",
    "ExecutionPlan",
    "ExecutionInputs",
    "JitError",
    "JitInputError",
    "JitInterpreterError",
    "OperationSupport",
    "OperationHandler",
    "ProgramRequirements",
    "PlanRegion",
    "PlanValue",
    "ProviderDecision",
    "ProviderPolicy",
    "RuntimeBindings",
    "Session",
    "TritonPointwiseBackend",
    "TritonPointwiseExecutable",
    "analyze_requirements",
    "check_interpreter_coverage",
]
