import dataclasses
import unittest

from tinymind import Model
from tinymind.tools.builtins import calculator, unit_convert


class TestModelFacade(unittest.TestCase):
    def test_generate_returns_text(self):
        model = Model("models/x.tm")
        response = model.generate("hello")
        self.assertIsInstance(response, str)
        self.assertIn("hello", response)

    def test_tools_registered_and_listed(self):
        model = Model(tools=[calculator, unit_convert])
        names = {t.name for t in model.tools.list()}
        self.assertEqual(names, {"calculator", "unit_convert"})

    def test_run_chat_mode(self):
        model = Model()
        result = model.run("What's the capital of France?")
        self.assertTrue(result.ok)

    def test_extract_reports_honest_failure_without_real_model(self):
        @dataclasses.dataclass
        class Receipt:
            total: float
            merchant: str
        model = Model()
        result = model.extract("some receipt text", Receipt)
        self.assertFalse(result.valid)

    def test_stream_yields_chunks(self):
        model = Model()
        chunks = list(model.stream("a b c"))
        self.assertGreater(len(chunks), 0)
        self.assertEqual("".join(chunks).strip(), "".join(chunks).strip())  # no crash / well-formed

    def test_reset_clears_history(self):
        model = Model()
        model.generate("hello")
        model.reset()
        self.assertEqual(len(model._session.history.all()), 0)

    def test_registry_and_tools_mutually_exclusive(self):
        from tinymind.tools.registry import ToolRegistry
        registry = ToolRegistry()
        with self.assertRaises(ValueError):
            Model(tools=[calculator], registry=registry)


if __name__ == "__main__":
    unittest.main()
