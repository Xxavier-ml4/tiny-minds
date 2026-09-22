import unittest

from tinymind.runtime.constraints import (
    ChoiceGrammar, JsonPrefixStateMachine, RegexGrammar, parse_structured_output,
)


class TestJsonSchemaConstraints(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {"room": {"type": "string"}, "brightness": {"type": "integer"}},
            "required": ["room", "brightness"],
        }

    def test_valid_json_parses(self):
        result = parse_structured_output('{"room": "bedroom", "brightness": 30}', self.schema)
        self.assertTrue(result.valid)
        self.assertEqual(result.value["brightness"], 30)
        self.assertFalse(result.repaired)

    def test_repairs_trailing_comma(self):
        result = parse_structured_output('{"room": "bedroom", "brightness": 30,}', self.schema)
        self.assertTrue(result.valid)
        self.assertTrue(result.repaired)

    def test_repairs_missing_closing_brace(self):
        result = parse_structured_output('{"room": "bedroom", "brightness": 30', self.schema)
        self.assertTrue(result.valid)
        self.assertTrue(result.repaired)

    def test_rejects_out_of_schema_value(self):
        result = parse_structured_output('{"room": "bedroom", "brightness": "thirty"}', self.schema)
        self.assertFalse(result.valid)

    def test_garbage_does_not_crash(self):
        result = parse_structured_output("not json at all {{{", self.schema)
        self.assertFalse(result.valid)


class TestJsonPrefixStateMachine(unittest.TestCase):
    def _feed_all(self, text):
        machine = JsonPrefixStateMachine()
        machine.feed(text)
        return machine

    def test_complete_object(self):
        m = self._feed_all('{"a": 1}')
        self.assertTrue(m.is_complete())
        self.assertTrue(m.is_valid_so_far())

    def test_incomplete_but_valid_prefix(self):
        m = self._feed_all('{"a": "bed')
        self.assertFalse(m.is_complete())
        self.assertTrue(m.is_valid_so_far())

    def test_trailing_comma_before_close_is_broken(self):
        m = self._feed_all('{"a": 1,}')
        self.assertFalse(m.is_valid_so_far())

    def test_extra_data_after_complete_value_is_broken(self):
        m = self._feed_all('{"a": 1} garbage')
        self.assertFalse(m.is_valid_so_far())

    def test_streamed_literal_true(self):
        machine = JsonPrefixStateMachine()
        for ch in "true":
            machine.feed(ch)
            self.assertTrue(machine.is_valid_so_far())
        self.assertTrue(machine.is_complete())

    def test_incremental_feed_matches_single_shot(self):
        text = '{"room": "bedroom", "brightness": 30}'
        incremental = JsonPrefixStateMachine()
        for ch in text:
            incremental.feed(ch)
        single_shot = self._feed_all(text)
        self.assertEqual(incremental.is_complete(), single_shot.is_complete())
        self.assertEqual(incremental.is_valid_so_far(), single_shot.is_valid_so_far())


class TestGrammars(unittest.TestCase):
    def test_choice_grammar(self):
        g = ChoiceGrammar(["heat", "cool", "auto"])
        self.assertTrue(g.matches("auto"))
        self.assertFalse(g.matches("warm"))
        self.assertTrue(g.is_valid_prefix("au"))
        self.assertFalse(g.is_valid_prefix("zz"))

    def test_regex_grammar_full_match(self):
        g = RegexGrammar(r"[A-Z]{2}\d{4}")
        self.assertTrue(g.matches("AB1234"))
        self.assertFalse(g.matches("AB123"))

    def test_regex_grammar_prefix_with_homogeneous_completion(self):
        # See RegexGrammar's docstring: it can find a completion when the
        # remaining pattern is homogeneous (all digits here).
        g = RegexGrammar(r"[A-Z]{2}\d{4}")
        self.assertTrue(g.is_valid_prefix("AB12"))
        self.assertTrue(g.is_valid_prefix("AB"))

    def test_regex_grammar_documented_heterogeneous_limitation(self):
        # Documented, expected miss: completing "A" needs one more letter
        # THEN four digits (heterogeneous), which the homogeneous-padding
        # probe does not search for. See RegexGrammar's docstring.
        g = RegexGrammar(r"[A-Z]{2}\d{4}")
        self.assertFalse(g.is_valid_prefix("A"))


if __name__ == "__main__":
    unittest.main()
