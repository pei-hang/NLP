from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from personal_assistant.memory import LongTermMemoryStore
from personal_assistant.tools import calculate_expression


class ToolsAndMemoryTest(unittest.TestCase):
    def test_calculator_expression(self) -> None:
        self.assertEqual(calculate_expression("256 × 1024"), "262144")
        self.assertEqual(calculate_expression("(10 + 2) / 3"), "4")

    def test_long_term_memory_extracts_preferences(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = LongTermMemoryStore(Path(tmpdir) / "memory.json")
            remembered = store.update_from_user_message("我平时喜欢喝咖啡，不爱喝茶，尽量给我推荐冷饮")
            self.assertTrue(any("咖啡" in item for item in remembered))
            self.assertTrue(any("茶" in item for item in remembered))
            self.assertTrue(any("冷饮" in item for item in remembered))
            self.assertFalse(any(item == "用户喜欢喝茶" for item in remembered))
            retrieved = store.format_for_prompt("推荐饮品")
            self.assertIn("咖啡", retrieved)
            self.assertIn("冷饮", retrieved)

    def test_memory_ignores_invalid_surrogate_characters(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = LongTermMemoryStore(path)
            store.update_from_user_message("我喜欢喝咖啡\udce5")
            reloaded = LongTermMemoryStore(path)
            self.assertIn("咖啡", reloaded.format_for_prompt("饮品"))

    def test_empty_memory_file_loads_as_empty_store(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            path.write_text("", encoding="utf-8")
            store = LongTermMemoryStore(path)
            self.assertEqual(store.entries, [])

    def test_latest_opposite_preference_wins(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = LongTermMemoryStore(path)
            store.update_from_user_message("我喜欢吃蛋糕")
            store.update_from_user_message("不喜欢吃蛋糕")
            memory_text = store.format_for_prompt("蛋糕")
            self.assertIn("用户不喜欢吃蛋糕", memory_text)
            self.assertNotIn("用户喜欢吃蛋糕", memory_text)

    def test_task_request_with_like_is_not_saved_as_memory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = LongTermMemoryStore(path)
            remembered = store.update_from_user_message("给我喜欢的人写一封信")
            self.assertEqual(remembered, [])
            self.assertEqual(store.entries, [])

    def test_self_disclosure_with_like_is_saved(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = LongTermMemoryStore(path)
            remembered = store.update_from_user_message("我喜欢一个人")
            self.assertEqual(remembered, ["用户喜欢一个人"])


if __name__ == "__main__":
    unittest.main()
