"""Reuse high-level templates and guarded selection plans across instances."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._template import StructuralTemplate


@dataclass(frozen=True)
class StructuralEntry:
    """Retain a numerical-data-free template and its current selection plans."""

    key: str
    template: StructuralTemplate
    manifest: dict[str, object]
    plans: dict[str, object]
    encoding: dict[str, object]

    @classmethod
    def create(
        cls,
        template: StructuralTemplate,
        manifest: dict[str, object],
        plans: dict[str, object],
    ) -> StructuralEntry:
        encoding = template.program.encode()
        identity = json.dumps(
            [encoding, template.metadata, manifest],
            sort_keys=True,
            allow_nan=False,
        ).encode()
        return cls(
            hashlib.blake2b(identity, digest_size=16).hexdigest(),
            template,
            manifest,
            plans,
            encoding,
        )

    def with_plans(self, plans: dict[str, object]) -> StructuralEntry:
        return replace(self, plans=plans)


# Entries contain IR and JSON metadata, never captured owners or material Tensors.
# A bounded process cache shares one source family across separately wrapped functions.
_MEMORY: OrderedDict[tuple[str | None, str], tuple[StructuralEntry, ...]] = (
    OrderedDict()
)
_MEMORY_LOCK = RLock()
_MAX_FAMILIES = 16
_MAX_VARIANTS = 16
_SCHEMA = 2


class StructuralCache:
    """Store guarded variants by source family, with optional disk persistence.

    Family entries are immutable snapshots. A compilation receives its own selection-plan object and publishes a replacement snapshot after preparation, so concurrent callers never mutate a shared Program or plan. Atomic replacement prevents partial files; concurrent writers may evict a variant, which only causes a later cache miss.
    """

    def __init__(self, directory: Path | None):
        self.directory = directory
        self._namespace = (
            None if directory is None else str(directory.resolve())
        )

    def _path(self, family: str) -> Path:
        assert self.directory is not None
        return self.directory / "structures" / family[:2] / (family + ".json")

    def _remember(
        self, family: str, entries: tuple[StructuralEntry, ...]
    ) -> None:
        identity = (self._namespace, family)
        with _MEMORY_LOCK:
            _MEMORY[identity] = entries
            _MEMORY.move_to_end(identity)
            while len(_MEMORY) > _MAX_FAMILIES:
                _MEMORY.popitem(last=False)

    def load(self, family: str) -> tuple[tuple[StructuralEntry, ...], str]:
        """Return available variants and the memory/disk/miss lookup layer."""
        identity = (self._namespace, family)
        with _MEMORY_LOCK:
            if identity in _MEMORY:
                _MEMORY.move_to_end(identity)
                return _MEMORY[identity], "memory"
        if self.directory is None:
            return (), "miss"
        from fhelium.ir import Program
        from ._template import StructuralTemplate

        try:
            payload = json.loads(self._path(family).read_text(encoding="utf-8"))
            if payload["schema"] != _SCHEMA or payload["family"] != family:
                return (), "miss"
            entries = tuple(
                StructuralEntry(
                    item["key"],
                    StructuralTemplate(
                        Program.decode(item["program"]),
                        item["metadata"],
                    ),
                    item["materials"],
                    item["plans"],
                    item["program"],
                )
                for item in payload["variants"]
            )
        except Exception:
            return (), "miss"
        self._remember(family, entries)
        return entries, "disk"

    def store(self, family: str, entry: StructuralEntry) -> None:
        """Publish an updated variant while retaining other observed guards."""
        prior, _ = self.load(family)
        entries = tuple(item for item in prior if item.key != entry.key)
        entries = (*entries, entry)[-_MAX_VARIANTS:]
        self._remember(family, entries)
        if self.directory is None:
            return
        payload = {
            "schema": _SCHEMA,
            "family": family,
            "variants": [
                {
                    "key": item.key,
                    "program": item.encoding,
                    "metadata": item.template.metadata,
                    "materials": item.manifest,
                    "plans": item.plans,
                }
                for item in entries
            ],
        }
        path = self._path(family)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, allow_nan=False)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=family + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(encoded)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
