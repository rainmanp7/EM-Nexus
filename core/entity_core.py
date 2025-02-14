# core/entity_core.py

import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from memory_store import MemoryStore
from core.learning_engine import LearningEngine
from core.entanglement_hub import EntanglementHub
from domains.math_module import MathModule
from domains.english_module import EnglishModule
from domains.python_module import PythonModule
from domains.science_module import ScienceModule
from core.holographic_memory import HolographicMemory  # Import HolographicMemory

class SuperEntity:
    def __init__(self, name, meta_entity=None, holographic_memory=None):
        self.name = name
        self.math_memory = MemoryStore(db_path="data/math.db")
        self.english_memory = MemoryStore(db_path="data/english.db")
        self.programming_memory = MemoryStore(db_path="data/programming.db")
        self.science_memory = MemoryStore(db_path="data/science.db")
        self.learning_engine = LearningEngine(self.math_memory)  # Default to math memory
        self.entanglement_hub = EntanglementHub(self.name)
        
        # Initialize holographic memory if not provided
        self.holographic_memory = holographic_memory if holographic_memory else HolographicMemory(memory_file=f"data/{name}_holographic_memory.npy")
        
        self.modules = {
            "math": MathModule(memory_dimensions=16384, memory_store=self.math_memory),
            "english": EnglishModule(memory_dimensions=16384, memory_store=self.english_memory),
            "python": PythonModule(self.learning_engine, memory_store=self.programming_memory),
            "science": ScienceModule(self.learning_engine, memory_store=self.science_memory),
        }
        self.meta_entity = meta_entity

    def process_task(self, domain, task_input):
        """
        Process a task using the appropriate module.
        """
        if domain in self.modules:
            result = self.modules[domain].process(task_input)
            if self.meta_entity:
                self.meta_entity.integrate_task_result(self.name, domain, result)
            return result
        else:
            raise ValueError(f"Domain '{domain}' not supported.")

    def run(self):
        """
        Run the entity and process a set of predefined tasks.
        """
        print(f"[{self.name}] Entity running...")
        tasks = [
            {"domain": "math", "input": {"type": "addition", "a": 5, "b": 3}},
            {"domain": "english", "input": "Learn the word 'emergence'"},
            {"domain": "python", "input": "Write a function to calculate factorial"},
            {"domain": "science", "input": "Solve a physics problem involving force"},
        ]
        for task in tasks:
            result = self.process_task(task["domain"], task["input"])
            print(f"Task result: {result}")