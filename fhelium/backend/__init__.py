"""Resolve and execute Program operations with registered implementations."""

from .distributed import (
    TorchBroadcastImplementation,
    TorchCiphertextAddAllReduceImplementation,
    TorchGenericAllReduceImplementation,
    TorchProcessGroupQueryImplementation,
    distributed_operation_contributions,
    prepare_ciphertext_add_combine,
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
    "TorchCiphertextAddAllReduceImplementation",
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
    "prepare_ciphertext_add_combine",
    "requested_implementation",
]
