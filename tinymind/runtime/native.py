"""The Python binding to the future native inference engine
(``native/include/tinymind.h``, ``native/src/*.cpp`` — interface stubs, not
a working engine yet; see ``STATUS.md``).

Follows Needle's own choice of ``ctypes`` over a compiled CPython extension
(needle-analysis.md section 4: "Python -> ctypes.CDLL -> C ABI") —
independently arrived at here for the same reason it's a reasonable choice
for anyone in this position: ``ctypes`` needs no build step for the Python
side and works identically across CPython versions, which matters for a
mobile-first project that wants the Python layer to stay as simple as
possible (brief section 17: "Start with the simplest portable option...
Do not make Python responsible for the actual hot inference loop").

This module cannot do anything real yet — there is no compiled
``libtinymind.{so,dylib,dll}`` to load, because ``native/src/*.cpp`` are
interface stubs, not a working engine (see their module-level comments and
``STATUS.md``). ``NativeEngine.load()`` raises immediately and explicitly
rather than pretending; ``ModelBackend`` implementations in this delivery
(``tinymind.model.backends.echo.EchoBackend``) do not use this module at
all. When a real ``libtinymind`` exists, a ``NativeBackend(ModelBackend)``
implementation goes in ``tinymind/model/backends/native.py`` and uses this
module — the file doesn't exist yet because the library it would bind to
doesn't either.
"""
from __future__ import annotations

import ctypes
import platform
from pathlib import Path


class NativeEngineError(RuntimeError):
    pass


class NativeLibraryNotFoundError(NativeEngineError):
    pass


def platform_tag() -> str:
    """The platform tag this process is running on, in the same style as
    ``native/README`` and ``android/README.md``'s ABI matrix (e.g.
    ``"linux-x86_64"``, ``"macos-arm64"``). Real and useful today even
    without a library to fetch — it's the tag a future fetch step would
    need to get right, and getting it wrong (serving a Linux glibc binary
    to an Android/Bionic target) is exactly the failure mode
    ``android/README.md`` documents from this project's own field testing.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}.get(machine, machine)
    if system == "linux" and "android" in platform.platform().lower():
        return f"android-{machine}"
    name = {"linux": "linux", "darwin": "macos", "windows": "windows"}.get(system, system)
    return f"{name}-{machine}"


def expected_library_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "tinymind.dll"
    if system == "darwin":
        return "libtinymind.dylib"
    return "libtinymind.so"


class NativeEngine:
    """Binds to ``libtinymind`` via ``ctypes`` once it exists. Every method
    below mirrors a function in ``native/include/tinymind.h``."""

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._context = None

    def load_library(self, path: str | Path | None = None) -> None:
        search_path = Path(path) if path else Path(expected_library_name())
        if not search_path.is_file():
            raise NativeLibraryNotFoundError(
                f"no native library at {search_path} — TinyMind's native engine is a "
                f"design-and-interface-stub in this delivery (see native/src/*.cpp and "
                f"STATUS.md), not a compiled binary. Platform tag for reference: "
                f"{platform_tag()!r}")
        self._lib = ctypes.CDLL(str(search_path))
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        # Mirrors native/include/tinymind.h. Left unimplemented until a real
        # library exists to introspect — guessing argtypes/restype for
        # functions with no real implementation behind them would be
        # actively misleading (ctypes would happily "succeed" at calling
        # into undefined behavior).
        raise NotImplementedError(
            "signature configuration is written against native/include/tinymind.h "
            "once native/src has a real implementation to link; see STATUS.md")
