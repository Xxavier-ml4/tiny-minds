"""``Module``: a minimal base class for organizing learnable parameters,
in the same spirit as (and much smaller than) ``torch.nn.Module`` — needed
for the same reason ``tinymind/model/tensor.py`` exists: no ML framework is
available in this environment. Not explicitly named in the brief's file
list because the brief assumes a framework provides this; documented here
for the same reason ``tensor.py`` documents itself.

A ``Module`` tracks its own parameters (``Tensor`` objects with
``requires_grad=True``) and any child modules assigned as attributes;
``parameters()``/``named_parameters()`` walk the whole tree. That's the
entire feature set — no hooks, no device placement, no serialization
(serialization is ``tinymind.model.checkpoint``'s job, working from
``named_parameters()``).
"""
from __future__ import annotations

from typing import Iterator

from tinymind.model.tensor import Tensor


class Module:
    def __init__(self) -> None:
        # Assigned via __setattr__ below; declared here so a subclass that
        # forgets to call super().__init__() fails loudly and immediately
        # instead of mysteriously later.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for name, param in self._parameters.items():
            yield f"{prefix}{name}", param
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix=f"{prefix}{name}.")

    def parameters(self) -> list[Tensor]:
        return [p for _name, p in self.named_parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def count_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")
