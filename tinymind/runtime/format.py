"""The ``.tm`` model file format.

Kept from the Needle `.cact` format (needle-analysis.md section 15): one
fixed-size binary header carrying the full architecture geometry, so a
single loader handles every size in the ``ModelConfig`` family with no
per-shape special-casing. Changed, deliberately: the tensor directory here
is **named and versioned**, not positional — see
``docs/architecture/tinymind-design.md`` section 9 for why (a nameless
directory is fragile the moment the architecture adds, removes, or reorders
a tensor across format versions).

Every integer read from the file (counts, offsets, sizes) is checked
against the actual file size *before* it is used to slice into the file, so
a truncated or adversarially-crafted file raises a specific
``ModelFormatError`` subclass rather than reading out of bounds or
crashing — per the engineering brief section 31 ("never trust tensor
offsets, tensor sizes, metadata lengths") and section 54 ("never hide
errors"; no bare ``assert`` is used for this).

File layout::

    offset  size  field
    0       4     magic = b"TM01"
    4       4     u32 header_length (bytes of the JSON metadata block that follows)
    8       N     UTF-8 JSON metadata block (architecture geometry + arbitrary
                   string metadata; see ``write_model``)
    8+N     4     u32 tensor_count
    8+N+4   ...   tensor_count directory entries, each:
                     u16 name_length, name (UTF-8)
                     u8  dtype_code (see ``_DTYPE_CODES``)
                     u8  ndim
                     ndim x u32 shape dims
                     u64 offset (from start of file)
                     u64 nbytes
                     u32 crc32 of the tensor bytes
    ...     ...   tensor bytes, each tensor's ``offset`` 32-byte aligned
"""
from __future__ import annotations

import dataclasses
import json
import struct
import zlib
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"TM01"
_ALIGNMENT = 32
_MAX_HEADER_LENGTH = 16 * 1024 * 1024  # 16 MiB of metadata JSON is already absurd; reject beyond that
_MAX_TENSOR_COUNT = 1_000_000
_MAX_NAME_LENGTH = 4096
_MAX_NDIM = 8

_DTYPE_CODES = {"float32": 0, "float16": 1, "bfloat16": 2, "int8": 3, "int4": 4,
                "int3": 5, "int2": 6, "int32": 7, "uint8": 8}
_DTYPE_NAMES = {v: k for k, v in _DTYPE_CODES.items()}


class ModelFormatError(Exception):
    """Base class for every ``.tm`` format problem."""


class UnsupportedFormatVersionError(ModelFormatError):
    pass


class ModelFileNotFoundError(ModelFormatError):
    pass


class CorruptModelFileError(ModelFormatError):
    pass


@dataclasses.dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    crc32: int


@dataclasses.dataclass
class ModelFile:
    metadata: dict[str, Any]
    tensors: dict[str, TensorEntry]
    _path: Path | None = None

    def read_tensor(self, name: str) -> bytes:
        if self._path is None:
            raise ModelFormatError("this ModelFile was not opened from a file (no tensor bytes to read)")
        entry = self.tensors.get(name)
        if entry is None:
            raise KeyError(f"no tensor named {name!r} in this model file "
                           f"(available: {sorted(self.tensors)})")
        with self._path.open("rb") as handle:
            file_size = handle.seek(0, 2)
            _check_bounds(entry.offset, entry.nbytes, file_size, context=f"tensor {name!r}")
            handle.seek(entry.offset)
            data = handle.read(entry.nbytes)
        crc = zlib.crc32(data)
        if crc != entry.crc32:
            raise CorruptModelFileError(
                f"tensor {name!r} failed its checksum (expected {entry.crc32:#010x}, "
                f"got {crc:#010x}) — the file is corrupt or was truncated")
        return data


def _check_bounds(offset: int, nbytes: int, file_size: int, *, context: str) -> None:
    """``offset``/``nbytes`` arrive as Python ints unpacked from a 64-bit
    field (``struct`` unpacks ``Q`` into an arbitrary-precision Python
    int). Unlike a C/C++ reader summing two ``uint64_t`` values, ``offset +
    nbytes`` below can never silently wrap around to a small value — Python
    integers don't overflow — so a maliciously huge pair of 64-bit fields
    is compared exactly and correctly rejected below, not aliased into an
    in-bounds-looking result. See ``tests/test_format.py``'s
    ``test_huge_offset_and_size_do_not_overflow`` for this checked
    directly against the largest values the field width allows.
    """
    if offset < 0 or nbytes < 0:
        raise CorruptModelFileError(f"{context}: negative offset or size (offset={offset}, nbytes={nbytes})")
    if offset > file_size:
        raise CorruptModelFileError(
            f"{context}: offset {offset} is beyond the end of the file ({file_size} bytes)")
    if offset + nbytes > file_size:
        raise CorruptModelFileError(
            f"{context}: offset {offset} + size {nbytes} = {offset + nbytes} exceeds the "
            f"file size ({file_size} bytes) — file is truncated or the directory is corrupt")


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)


def write_model(path: str | Path, *, metadata: dict[str, Any],
                tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    """Write a ``.tm`` file.

    ``tensors`` maps a tensor name to ``(dtype, shape, raw_bytes)``. Callers
    are responsible for having already serialized ``raw_bytes`` in the
    layout implied by ``dtype``/``shape`` (this function does not know how
    to encode a tensor's numeric values — that's the quantization/export
    subsystem's job; this module only knows how to pack and unpack bytes
    safely, per ``docs/architecture/tinymind-design.md`` section 9).
    """
    path = Path(path)
    for name, (dtype, shape, _) in tensors.items():
        if len(name.encode("utf-8")) > _MAX_NAME_LENGTH:
            raise ModelFormatError(f"tensor name {name!r} exceeds {_MAX_NAME_LENGTH} bytes")
        if dtype not in _DTYPE_CODES:
            raise ModelFormatError(f"tensor {name!r} has unsupported dtype {dtype!r}; "
                                   f"supported: {sorted(_DTYPE_CODES)}")
        if len(shape) > _MAX_NDIM:
            raise ModelFormatError(f"tensor {name!r} has {len(shape)} dims, max is {_MAX_NDIM}")

    metadata_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    if len(metadata_bytes) > _MAX_HEADER_LENGTH:
        raise ModelFormatError(f"metadata block is {len(metadata_bytes)} bytes, "
                               f"exceeds the {_MAX_HEADER_LENGTH} byte limit")

    directory_entries = []
    for name, (dtype, shape, raw_bytes) in tensors.items():
        directory_entries.append((name, dtype, shape, raw_bytes, zlib.crc32(raw_bytes)))

    # First pass: compute the directory's fixed size so we know where tensor
    # bytes start; a directory entry's size depends only on name length and
    # ndim, both already known.
    directory_size = 4  # tensor_count
    for name, _dtype, shape, _raw, _crc in directory_entries:
        directory_size += 2 + len(name.encode("utf-8")) + 1 + 1 + 4 * len(shape) + 8 + 8 + 4

    header_start = 8 + len(metadata_bytes)
    cursor = _align_up(header_start + directory_size)
    placed: list[tuple[str, str, tuple[int, ...], bytes, int, int]] = []
    for name, dtype, shape, raw_bytes, crc in directory_entries:
        offset = cursor
        placed.append((name, dtype, shape, raw_bytes, offset, crc))
        cursor = _align_up(offset + len(raw_bytes))

    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", len(metadata_bytes)))
        handle.write(metadata_bytes)
        handle.write(struct.pack("<I", len(placed)))
        for name, dtype, shape, raw_bytes, offset, crc in placed:
            name_bytes = name.encode("utf-8")
            handle.write(struct.pack("<H", len(name_bytes)))
            handle.write(name_bytes)
            handle.write(struct.pack("<BB", _DTYPE_CODES[dtype], len(shape)))
            for dim in shape:
                handle.write(struct.pack("<I", dim))
            handle.write(struct.pack("<QQI", offset, len(raw_bytes), crc))
        for name, dtype, shape, raw_bytes, offset, crc in placed:
            handle.seek(offset)
            handle.write(raw_bytes)


def _read_exact(handle: BinaryIO, n: int, *, context: str) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise CorruptModelFileError(
            f"unexpected end of file while reading {context} "
            f"(wanted {n} bytes, got {len(data)})")
    return data


def read_model(path: str | Path) -> ModelFile:
    path = Path(path)
    try:
        file_size = path.stat().st_size
    except FileNotFoundError as exc:
        raise ModelFileNotFoundError(f"no such file: {path}") from exc
    with path.open("rb") as handle:
        magic = _read_exact(handle, 4, context="magic")
        if magic != MAGIC:
            raise UnsupportedFormatVersionError(
                f"unrecognized magic {magic!r}; TinyMind reads {MAGIC!r} ('TM01') only — "
                "this is not a .tm file, or it is a format version this reader does not "
                "support")

        (header_length,) = struct.unpack("<I", _read_exact(handle, 4, context="header length"))
        if header_length > _MAX_HEADER_LENGTH:
            raise CorruptModelFileError(
                f"declared metadata length {header_length} exceeds the sanity limit "
                f"({_MAX_HEADER_LENGTH} bytes); refusing to read further")
        _check_bounds(8, header_length, file_size, context="metadata block")
        metadata_bytes = _read_exact(handle, header_length, context="metadata block")
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptModelFileError(f"metadata block is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise CorruptModelFileError(
                f"metadata block must decode to a JSON object, got {type(metadata).__name__}")

        (tensor_count,) = struct.unpack("<I", _read_exact(handle, 4, context="tensor count"))
        if tensor_count > _MAX_TENSOR_COUNT:
            raise CorruptModelFileError(
                f"declared tensor count {tensor_count} exceeds the sanity limit "
                f"({_MAX_TENSOR_COUNT}); refusing to read further")

        tensors: dict[str, TensorEntry] = {}
        for i in range(tensor_count):
            (name_length,) = struct.unpack("<H", _read_exact(handle, 2, context=f"tensor[{i}] name length"))
            if name_length > _MAX_NAME_LENGTH:
                raise CorruptModelFileError(
                    f"tensor[{i}] name length {name_length} exceeds the sanity limit "
                    f"({_MAX_NAME_LENGTH})")
            name_bytes = _read_exact(handle, name_length, context=f"tensor[{i}] name")
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CorruptModelFileError(f"tensor[{i}] name is not valid UTF-8: {exc}") from exc
            if name in tensors:
                raise CorruptModelFileError(f"duplicate tensor name {name!r} in directory")

            dtype_code, ndim = struct.unpack("<BB", _read_exact(handle, 2, context=f"tensor[{i}] dtype/ndim"))
            if dtype_code not in _DTYPE_NAMES:
                raise CorruptModelFileError(f"tensor {name!r} has unknown dtype code {dtype_code}")
            if ndim > _MAX_NDIM:
                raise CorruptModelFileError(f"tensor {name!r} declares {ndim} dims, max is {_MAX_NDIM}")

            shape = []
            for _ in range(ndim):
                (dim,) = struct.unpack("<I", _read_exact(handle, 4, context=f"tensor {name!r} shape"))
                shape.append(dim)

            offset, nbytes, crc = struct.unpack("<QQI", _read_exact(
                handle, 20, context=f"tensor {name!r} offset/size/checksum"))
            _check_bounds(offset, nbytes, file_size, context=f"tensor {name!r} directory entry")

            tensors[name] = TensorEntry(name=name, dtype=_DTYPE_NAMES[dtype_code],
                                        shape=tuple(shape), offset=offset,
                                        nbytes=nbytes, crc32=crc)

        directory_end = handle.tell()

    _check_no_tensor_points_into_header(tensors, directory_end)
    _check_no_tensor_overlap(tensors)

    return ModelFile(metadata=metadata, tensors=tensors, _path=path)


def _check_no_tensor_points_into_header(tensors: dict[str, TensorEntry], directory_end: int) -> None:
    """Brief section 16/28's "malicious model files" hardening: a tensor
    whose declared offset falls before the end of the header/directory is
    already-in-bounds by the plain file-size check in ``_check_bounds``,
    but it's still wrong — it would read metadata/directory bytes as if
    they were tensor data. Reject it explicitly rather than silently
    returning header bytes as weights.
    """
    for name, entry in tensors.items():
        if entry.nbytes > 0 and entry.offset < directory_end:
            raise CorruptModelFileError(
                f"tensor {name!r} at offset {entry.offset} points into the header/directory "
                f"region (which ends at byte {directory_end}) instead of the tensor data region")


def _check_no_tensor_overlap(tensors: dict[str, TensorEntry]) -> None:
    """Two tensors whose byte ranges overlap can't both be legitimate —
    reject the file rather than silently letting one tensor's read return
    bytes that are also part of another tensor. A sweep over entries sorted
    by offset (O(n log n)) rather than checking every pair (O(n^2)), which
    matters once ``_MAX_TENSOR_COUNT`` (1,000,000) is taken seriously as an
    upper bound to actually defend, not just document.
    """
    ranges = sorted(((entry.offset, entry.offset + entry.nbytes, name)
                     for name, entry in tensors.items() if entry.nbytes > 0))
    for i in range(1, len(ranges)):
        prev_start, prev_end, prev_name = ranges[i - 1]
        start, _end, name = ranges[i]
        if start < prev_end:
            raise CorruptModelFileError(
                f"tensor {name!r} (offset {start}) overlaps tensor {prev_name!r} "
                f"(byte range [{prev_start}, {prev_end}))")
