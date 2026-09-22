import os
import tempfile
import unittest

from tinymind.memory import LongTermMemory, MemoryEntry, ShortTermMemory, retrieve_relevant


class TestShortTermMemory(unittest.TestCase):
    def test_bounded_and_keeps_most_recent(self):
        stm = ShortTermMemory(max_entries=3)
        for i in range(5):
            stm.add(MemoryEntry(content=f"message {i}", source="user"))
        contents = [e.content for e in stm.all()]
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents, ["message 2", "message 3", "message 4"])

    def test_clear(self):
        stm = ShortTermMemory()
        stm.add(MemoryEntry(content="x", source="user"))
        stm.clear()
        self.assertEqual(len(stm.all()), 0)

    def test_scope_filtering(self):
        stm = ShortTermMemory()
        stm.add(MemoryEntry(content="a", source="user", scope="conversation"))
        stm.add(MemoryEntry(content="b", source="user", scope="profile"))
        self.assertEqual(len(stm.all(scope="profile")), 1)


class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".db")

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_persists_across_reopen(self):
        ltm = LongTermMemory(self.path)
        ltm.add(MemoryEntry(content="a fact worth keeping", source="user"))
        ltm.close()
        reopened = LongTermMemory(self.path)
        self.assertEqual(len(reopened.all()), 1)
        reopened.close()

    def test_delete(self):
        ltm = LongTermMemory(self.path)
        entry_id = ltm.add(MemoryEntry(content="temporary", source="user"))
        ltm.delete(entry_id)
        self.assertIsNone(ltm.get(entry_id))
        ltm.close()


class TestRetrieval(unittest.TestCase):
    def test_retrieves_most_relevant_entry(self):
        stm = ShortTermMemory()
        stm.add(MemoryEntry(content="user prefers dark mode in the settings", source="user"))
        stm.add(MemoryEntry(content="user is planning a hiking trip next month", source="user"))
        stm.add(MemoryEntry(content="user asked about hiking trail recommendations", source="user"))
        results = retrieve_relevant("what trip is the user planning", stm, k=1)
        self.assertIn("hiking trip", results[0].content)

    def test_never_returns_more_than_k(self):
        stm = ShortTermMemory()
        for i in range(10):
            stm.add(MemoryEntry(content=f"note number {i} about topic", source="user"))
        results = retrieve_relevant("topic", stm, k=3)
        self.assertLessEqual(len(results), 3)

    def test_scope_restriction(self):
        stm = ShortTermMemory()
        stm.add(MemoryEntry(content="profile fact about hiking", source="user", scope="profile"))
        stm.add(MemoryEntry(content="conversation note about hiking", source="user", scope="conversation"))
        results = retrieve_relevant("hiking", stm, k=5, scope="profile")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].scope, "profile")


if __name__ == "__main__":
    unittest.main()
