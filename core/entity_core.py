import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# Now you can use absolute imports

from memory_store import MemoryStore
from core.learning_engine import LearningEngine
from core.entanglement_hub import EntanglementHub
from domains.math_module import MathModule
from domains.english_module import EnglishModule
from domains.python_module import PythonModule

class SuperEntity:
    def __init__(self, name, meta_entity=None, holographic_memory=None):
        self.name = name
        self.memory = MemoryStore(db_path="data/entity_memory.db")
        self.learning_engine = LearningEngine(self.memory)
        self.entanglement_hub = EntanglementHub(self.name)
        self.modules = {
            "math": MathModule(memory_dimensions=16384),  # Correct initialization
            "english": EnglishModule(memory_dimensions=16384),  # Correct initialization
            "python": PythonModule(self.learning_engine),  # LearningEngine is passed here
        }
        self.meta_entity = meta_entity  # Link to meta-entity, if applicable
        self.holographic_memory = holographic_memory  # Link to holographic memory, if applicable

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
    from main import main  # Move import inside the if __name__ == "__main__" block
    main()
