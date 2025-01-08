from memory_store import MemoryStore
from learning_engine import LearningEngine
from entanglement_hub import EntanglementHub
from domains.math_module import MathModule
from domains.english_module import EnglishModule
from domains.python_module import PythonModule

class SuperEntity:
    def __init__(self, name, meta_entity=None):
        self.name = name
        self.memory = MemoryStore(db_path="data/entity_memory.db")
        self.learning_engine = LearningEngine(self.memory)
        self.entanglement_hub = EntanglementHub(self.name)
        self.modules = {
            "math": MathModule(self.learning_engine),
            "english": EnglishModule(self.learning_engine),
            "python": PythonModule(self.learning_engine),
        }
        self.meta_entity = meta_entity  # Link to meta-entity, if applicable

    def process_task(self, domain, task_input):
        if domain in self.modules:
            result = self.modules[domain].process(task_input)
            if self.meta_entity:
                self.meta_entity.integrate_task_result(self.name, domain, result)
            return result
        else:
            raise ValueError(f"Domain '{domain}' not supported.")

    def run(self):
        print(f"[{self.name}] Entity running...")
        tasks = [
            {"domain": "math", "input": {"type": "addition", "a": 5, "b": 3}},
            {"domain": "english", "input": "Learn the word 'emergence'"},
            {"domain": "python", "input": "Write a function to calculate factorial"},
        ]
        for task in tasks:
            result = self.process_task(task["domain"], task["input"])
            print(f"Task result: {result}")

if __name__ == "__main__":
    entity = SuperEntity("SuperEntity1")
    entity.run()
