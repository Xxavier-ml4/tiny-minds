import unittest

from tinymind.runtime.native import NativeLibraryNotFoundError, expected_library_name, platform_tag


class TestNativeBindingStub(unittest.TestCase):
    def test_platform_tag_is_nonempty_string(self):
        tag = platform_tag()
        self.assertIsInstance(tag, str)
        self.assertIn("-", tag)  # "<os>-<arch>" shape

    def test_expected_library_name_matches_platform_convention(self):
        name = expected_library_name()
        self.assertTrue(name.endswith((".so", ".dylib", ".dll")))

    def test_load_missing_library_fails_honestly(self):
        from tinymind.runtime.native import NativeEngine
        engine = NativeEngine()
        with self.assertRaises(NativeLibraryNotFoundError):
            engine.load_library("/nonexistent/libtinymind.so")


if __name__ == "__main__":
    unittest.main()
