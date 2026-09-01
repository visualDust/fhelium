"""Compose Backend operation resolution and workspace linking."""

from __future__ import annotations

from ..._pipeline import (
    Pipeline,
)

from fhelium.backend.execution import OperationBackend


from ._link_program import LinkProgramPass
from ._bind_ckks_keys import BindCkksKeysPass
from ._initialize_resources import InitializeResourceBindingsPass
from ._materialize_resources import MaterializeResourcesPass
from ._resolve_operations import ResolveBackendOperationsPass


def backend_linking_pipeline(
    backend: OperationBackend,
    *,
    in_place: bool = False,
) -> Pipeline:
    """Return the standard Backend linking sequence as an editable Pipeline."""

    workspace = backend.workspace
    resources = workspace.named_resources
    key_binding = (
        () if not workspace.keys else (BindCkksKeysPass(workspace.keys),)
    )
    materialization = (
        ()
        if workspace.materializer is None
        else (MaterializeResourcesPass(workspace.materializer),)
    )
    return Pipeline(
        (
            ResolveBackendOperationsPass(
                backend.registry,
                in_place=in_place,
            ),
            InitializeResourceBindingsPass(resources),
            *key_binding,
            *materialization,
            LinkProgramPass(workspace.material_overrides),
        )
    )


__all__ = ["backend_linking_pipeline"]
