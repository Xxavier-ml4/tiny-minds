"""Tool permission tags, per the engineering brief section 4.

Every registered tool declares zero or more capability tags. The executor
(``tinymind/tools/executor.py``) refuses to run a ``DESTRUCTIVE`` or
``SENSITIVE`` tool unless the caller passes explicit confirmation — the
model itself is never in a position to grant that confirmation, only the
application embedding TinyMind is.
"""
from __future__ import annotations

import dataclasses
import enum


class Capability(enum.Flag):
    READ_ONLY = enum.auto()
    LOCAL_WRITE = enum.auto()
    NETWORK = enum.auto()
    SENSITIVE = enum.auto()
    DESTRUCTIVE = enum.auto()


CONFIRMATION_REQUIRED = Capability.SENSITIVE | Capability.DESTRUCTIVE


@dataclasses.dataclass(frozen=True)
class ToolPermissions:
    capabilities: Capability = Capability.READ_ONLY
    requires_confirmation: bool | None = None
    """None = derive from capabilities (True iff SENSITIVE or DESTRUCTIVE is
    set); an explicit True/False overrides that default."""

    @property
    def needs_confirmation(self) -> bool:
        if self.requires_confirmation is not None:
            return self.requires_confirmation
        return bool(self.capabilities & CONFIRMATION_REQUIRED)

    def describe(self) -> str:
        names = [flag.name for flag in Capability if flag in self.capabilities and flag.name]
        return "+".join(names) if names else "none"
