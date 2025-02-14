# entity_controller.py
# diagnostic_test.py

import sys
import os
import sqlite3

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from domains.math_module import MathModule
from domains.english_module import EnglishModule
from programming_module import ProgrammingModule
from core.learning_engine import LearningEngine
from memory_store import MemoryStore
from core.holographic_memory import HolographicMemory  # Import HolographicMemory

class EntityController:
    def __init__(self):
        # Initialize the LearningEngine with a MemoryStore and HolographicMemory
        self.holographic_memory = HolographicMemory(dimensions=16384)  # Add holographic memory
        learning_engine = LearningEngine(MemoryStore("data/entity_memory.db"))

        # Initialize modules with required arguments
        self.math_module = MathModule()
        self.english_module = EnglishModule()
        self.programming_module = ProgrammingModule(learning_engine)

    def process_task(self, task_type, input_data):
        """
        Process a task using the appropriate module.
        :param task_type: Type of the task ('math', 'english', 'programming').
        :param input_data: Task-specific input.
        """
        if task_type == "math":
            return self.math_module.retrieve_solution(input_data)
        elif task_type == "english":
            return self.english_module.retrieve_meaning(input_data)
        elif task_type == "programming":
            return self.programming_module.retrieve_description(input_data)

# Example usage
if __name__ == "__main__":
    controller = EntityController()
    print(controller.process_task("math", "5 + 5"))