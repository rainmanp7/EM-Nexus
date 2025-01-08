import unittest
from entity_core import SuperEntity

class TestEntity(unittest.TestCase):
    def test_process_math_task(self):
        entity = SuperEntity("TestEntity")
        task = {"domain": "math", "input": {"type": "addition", "a": 3, "b": 7}}
        result = entity.process_task(task["domain"], task["input"])
        self.assertEqual(result, 10)

    def test_process_english_task(self):
        entity = SuperEntity("TestEntity")
        task = {"domain": "english", "input": "Learn the word 'emergence'"}
        result = entity.process_task(task["domain"], task["input"])
        self.assertIn("Learned", result)

    def test_process_python_task(self):
        entity = SuperEntity("TestEntity")
        task = {"domain": "python", "input": "Write a function to calculate factorial"}
        result = entity.process_task(task["domain"], task["input"])
        self.assertIn("def factorial", result)
