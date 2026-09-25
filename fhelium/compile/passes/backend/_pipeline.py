"""Compose Backend operation resolution and workspace linking."""

from __future__ import annotations


from ..._pipeline import (
    Pipeline,
)

from fhelium.backend.execution import OperationBackend


from ._link_program import LinkProgramPass
from ._initialize_resources import InitializeResourceBindingsPass
from ._materialize_resources import MaterializeResourcesPass
from ._resolve_operations import ResolveBackendOperationsPass
from ._resolve_tensor_placeholders import ResolveTensorPlaceholdersPass


def backend_linking_pipeline(
    backend: OperationBackend,
    *,
    in_place: bool = False,
) -> Pipeline:
    """Return the standard Backend linking sequence as an editable Pipeline."""

    workspace = backend.workspace
    resources = workspace.named_resources
    materialization = (
        ()
        if workspace.materializer is None
        else (MaterializeResourcesPass(workspace.materializer),)
    )
    return Pipeline(
        (
            ResolveTensorPlaceholdersPass(),
            ResolveBackendOperationsPass(
                backend.registry,
                in_place=in_place,
            ),
            InitializeResourceBindingsPass(resources),
            *materialization,
            LinkProgramPass(),
        )
    )


__all__ = ["backend_linking_pipeline"]
