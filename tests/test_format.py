import json
import os
import struct
import tempfile
import unittest

import tinymind.runtime.format as fmt


class TestModelFormat(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".tm")
        self.tensors = {
            "embed.weight": ("float32", (10, 8), bytes(range(80))),
            "layer0.attn.q.weight": ("int8", (8, 8), bytes(range(64))),
        }
        fmt.write_model(self.path, metadata={"architecture": "tinymind-test", "hidden_size": 8},
                        tensors=self.tensors)

    def tearDown(self):
        for path in (self.path,):
            if os.path.exists(path):
                os.remove(path)

    def test_round_trip(self):
        model_file = fmt.read_model(self.path)
        self.assertEqual(model_file.metadata["architecture"], "tinymind-test")
        self.assertEqual(set(model_file.tensors), set(self.tensors))
        for name, (_dtype, _shape, raw) in self.tensors.items():
            self.assertEqual(model_file.read_tensor(name), raw)

    def test_tensor_shape_and_dtype_preserved(self):
        model_file = fmt.read_model(self.path)
        entry = model_file.tensors["embed.weight"]
        self.assertEqual(entry.dtype, "float32")
        self.assertEqual(entry.shape, (10, 8))

    def test_bad_magic_raises_unsupported_version(self):
        bad_path = tempfile.mktemp(suffix=".tm")
        with open(self.path, "rb") as f:
            data = f.read()
        with open(bad_path, "wb") as f:
            f.write(b"XXXX" + data[4:])
        try:
            with self.assertRaises(fmt.UnsupportedFormatVersionError):
                fmt.read_model(bad_path)
        finally:
            os.remove(bad_path)

    def test_truncated_file_raises_corrupt(self):
        trunc_path = tempfile.mktemp(suffix=".tm")
        with open(self.path, "rb") as f:
            data = f.read()
        with open(trunc_path, "wb") as f:
            f.write(data[: len(data) // 2])
        try:
            with self.assertRaises(fmt.CorruptModelFileError):
                fmt.read_model(trunc_path)
        finally:
            os.remove(trunc_path)

    def test_out_of_bounds_tensor_offset_raises_corrupt(self):
        tampered_path = tempfile.mktemp(suffix=".tm")
        with open(self.path, "rb") as f:
            data = bytearray(f.read())
        header_length = struct.unpack("<I", bytes(data[4:8]))[0]
        cursor = 8 + header_length + 4  # skip magic, header_length, metadata, tensor_count
        name_length = struct.unpack("<H", bytes(data[cursor:cursor + 2]))[0]
        cursor += 2 + name_length
        _dtype_code, ndim = struct.unpack("<BB", bytes(data[cursor:cursor + 2]))
        cursor += 2 + 4 * ndim
        struct.pack_into("<Q", data, cursor, 999_999_999)  # bogus offset
        with open(tampered_path, "wb") as f:
            f.write(bytes(data))
        try:
            with self.assertRaises(fmt.CorruptModelFileError):
                fmt.read_model(tampered_path)
        finally:
            os.remove(tampered_path)

    def test_corrupted_tensor_bytes_fail_checksum(self):
        tampered_path = tempfile.mktemp(suffix=".tm")
        with open(self.path, "rb") as f:
            data = bytearray(f.read())
        data[-1] ^= 0xFF  # flip a bit inside the last tensor's bytes
        with open(tampered_path, "wb") as f:
            f.write(bytes(data))
        try:
            model_file = fmt.read_model(tampered_path)
            with self.assertRaises(fmt.CorruptModelFileError):
                model_file.read_tensor("layer0.attn.q.weight")
        finally:
            os.remove(tampered_path)

    def test_duplicate_tensor_name_rejected(self):
        # Hand-craft a minimal file with two directory entries sharing a name.
        import zlib
        metadata = json.dumps({}).encode("utf-8")
        raw = b"\x00" * 4
        crc = zlib.crc32(raw)
        dup_path = tempfile.mktemp(suffix=".tm")
        with open(dup_path, "wb") as f:
            f.write(fmt.MAGIC)
            f.write(struct.pack("<I", len(metadata)))
            f.write(metadata)
            f.write(struct.pack("<I", 2))  # tensor_count = 2, both named "x"
            offset = 8 + len(metadata) + 4 + 2 * (2 + 1 + 1 + 0 + 8 + 8 + 4)
            for _ in range(2):
                f.write(struct.pack("<H", 1))
                f.write(b"x")
                f.write(struct.pack("<BB", 0, 0))  # float32, ndim=0 (scalar)
                f.write(struct.pack("<QQI", offset, len(raw), crc))
            f.seek(offset)
            f.write(raw)
        try:
            with self.assertRaises(fmt.CorruptModelFileError):
                fmt.read_model(dup_path)
        finally:
            os.remove(dup_path)

    def test_unknown_dtype_rejected(self):
        with self.assertRaises(fmt.ModelFormatError):
            fmt.write_model(tempfile.mktemp(suffix=".tm"), metadata={},
                            tensors={"x": ("not_a_real_dtype", (1,), b"\x00\x00\x00\x00")})


class TestFormatHardening(unittest.TestCase):
    """Phase 3A hardening additions (brief sections 16/17/28): tensors may
    not overlap, may not point into the header/directory region, and a
    maximally-large 64-bit offset/size pair must not be mistaken for a
    small in-bounds value."""

    @staticmethod
    def _craft_file(path, tensor_entries, extra_bytes_after_directory=64):
        """``tensor_entries``: list of (name, offset, nbytes) — crafted
        directly, bypassing write_model, so offsets can be deliberately
        invalid in ways write_model itself would never produce. The file is
        padded to comfortably fit every entry's ``offset + nbytes`` (except
        deliberately-absurd huge values, which are left alone so the test
        that uses them stays a small file on purpose)."""
        import zlib
        metadata = json.dumps({}).encode("utf-8")
        needed_end = max((offset + nbytes for _name, offset, nbytes in tensor_entries
                         if offset + nbytes < 10**9), default=0)
        with open(path, "wb") as f:
            f.write(fmt.MAGIC)
            f.write(struct.pack("<I", len(metadata)))
            f.write(metadata)
            f.write(struct.pack("<I", len(tensor_entries)))
            for name, offset, nbytes in tensor_entries:
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<BB", 0, 0))  # float32, ndim=0 (scalar; shape irrelevant here)
                crc = zlib.crc32(b"\x00" * nbytes) if nbytes < 10_000_000 else 0
                f.write(struct.pack("<QQI", offset, nbytes, crc))
            directory_end = f.tell()
            pad_to = max(directory_end + extra_bytes_after_directory, needed_end)
            f.write(b"\x00" * (pad_to - directory_end))

    def test_tensor_overlap_rejected(self):
        path = tempfile.mktemp(suffix=".tm")
        try:
            self._craft_file(path, [("a", 100, 20), ("b", 110, 20)])  # [100,120) and [110,130) overlap
            with self.assertRaises(fmt.CorruptModelFileError) as ctx:
                fmt.read_model(path)
            self.assertIn("overlap", str(ctx.exception))
        finally:
            os.remove(path)

    def test_adjacent_non_overlapping_tensors_accepted(self):
        path = tempfile.mktemp(suffix=".tm")
        try:
            self._craft_file(path, [("a", 100, 20), ("b", 120, 20)])  # [100,120) and [120,140) touch, don't overlap
            model_file = fmt.read_model(path)  # must not raise
            self.assertEqual(set(model_file.tensors), {"a", "b"})
        finally:
            os.remove(path)

    def test_tensor_pointing_into_directory_rejected(self):
        path = tempfile.mktemp(suffix=".tm")
        try:
            # Offset 4 is inside the magic+header region, nowhere near
            # where real tensor data would ever legitimately start.
            self._craft_file(path, [("a", 4, 8)])
            with self.assertRaises(fmt.CorruptModelFileError) as ctx:
                fmt.read_model(path)
            self.assertIn("header/directory", str(ctx.exception))
        finally:
            os.remove(path)

    def test_huge_offset_and_size_do_not_overflow(self):
        path = tempfile.mktemp(suffix=".tm")
        try:
            # The largest values a u64 field can hold — in a naive C
            # uint64_t sum this would wrap around; in Python it must not,
            # and must be rejected as (very) out of bounds.
            huge = (1 << 64) - 1
            self._craft_file(path, [("a", huge, huge)])
            with self.assertRaises(fmt.CorruptModelFileError) as ctx:
                fmt.read_model(path)
            self.assertIn(str(huge), str(ctx.exception))
        finally:
            os.remove(path)

    def test_missing_file_raises_clean_error_not_raw_filenotfounderror(self):
        with self.assertRaises(fmt.ModelFileNotFoundError):
            fmt.read_model("/this/path/does/not/exist.tm")
        # ModelFileNotFoundError must still be a ModelFormatError, so
        # existing "except ModelFormatError" call sites (tinymind.cli,
        # tinymind.model.tm_export) keep working without needing to know
        # about this specific subclass.
        self.assertTrue(issubclass(fmt.ModelFileNotFoundError, fmt.ModelFormatError))


if __name__ == "__main__":
    unittest.main()
