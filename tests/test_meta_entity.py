import unittest
from meta_entity_core import MetaEntity
from entity_core import SuperEntity

class TestMetaEntity(unittest.TestCase):
    def setUp(self):
        self.meta_entity = MetaEntity("TestMetaEntity")
        self.entity1 = SuperEntity("Entity1", self.meta_entity)
        self.entity2 = SuperEntity("Entity2", self.meta_entity)
        self.meta_entity.register_entity(self.entity1)
        self.meta_entity.register_entity(self.entity2)

    def test_register_entity(self):
        self.assertEqual(len(self.meta_entity.entities), 2)

    def test_integrate_task_result(self):
        self.meta_entity.integrate_task_result("Entity1", "math", "10 + 20 = 30")
        knowledge = self.meta_entity.memory.retrieve_experiences()
        self.assertTrue(knowledge)

    def test_process_meta_task(self):
        meta_task = {
            "description": "Collaborative task test",
            "sub_tasks": [
                {"domain": "math", "input": {"type": "addition", "a": 5, "b": 15}},
            ],
        }
        results = self.meta_entity.process_meta_task(meta_task)
        self.assertTrue(results)

if __name__ == "__main__":
    unittest.main()
