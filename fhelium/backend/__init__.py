"""Resolve and execute Program operations with registered implementations."""

from .distributed import (
    TorchBroadcastImplementation,
    TorchCiphertextAllReduceImplementation,
    TorchGenericAllReduceImplementation,
    TorchProcessGroupQueryImplementation,
    distributed_operation_contributions,
    ProcessGroupExecutionResource,
)
from .execution import (
    OperationBackend,
    OperationDispatch,
    ProgramDispatchTable,
    ProgramExecutable,
)
from .memory import (
    DEVICE_RESOURCE_KIND,
    TorchMemoryTransferImplementation,
)
from .implementation import (
    ImplementationRegistry,
    OperationImplementation,
    OperationImplementationRegistry,
    OperationInvocation,
    requested_implementation,
)
from .resources import ResourceMaterializer
from .workspace import BackendWorkspace

__all__ = [
    "BackendWorkspace",
    "TorchBroadcastImplementation",
    "TorchCiphertextAllReduceImplementation",
    "DEVICE_RESOURCE_KIND",
    "TorchGenericAllReduceImplementation",
    "ImplementationRegistry",
    "TorchMemoryTransferImplementation",
    "OperationBackend",
    "OperationDispatch",
    "OperationImplementation",
    "OperationImplementationRegistry",
    "OperationInvocation",
    "ProgramDispatchTable",
    "ProgramExecutable",
    "ResourceMaterializer",
    "ProcessGroupExecutionResource",
    "TorchProcessGroupQueryImplementation",
    "distributed_operation_contributions",
    "requested_implementation",
]
