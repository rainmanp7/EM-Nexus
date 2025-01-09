# core/meta_entity_core.py
# diagnostic_test.py

import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from memory_store import MemoryStore
from core.meta_learning import MetaLearning
from core.entity_core import SuperEntity

class MetaEntity:
    def __init__(self, name):
        self.name = name
        self.memory = MemoryStore(db_path="data/meta_memory.db")
        self.meta_learning = MetaLearning()
        self.entities = []  # List of entities managed by the meta-entity

    def register_entity(self, entity):
        """Register an individual entity with the meta-entity."""
        self.entities.append(entity)
        print(f"[{self.name}] Registered entity: {entity.name}")

    def integrate_task_result(self, entity_name, domain, result):
        """Store and analyze task results from an individual entity."""
        self.memory.store_knowledge(f"Result from {entity_name}", result, domain)
        print(f"[{self.name}] Integrated result from {entity_name} in domain '{domain}': {result}")

    def process_meta_task(self, meta_task):
        """Distribute meta-tasks among entities and coordinate results."""
        print(f"[{self.name}] Processing meta-task: {meta_task['description']}")
        results = []
        for task in meta_task["sub_tasks"]:
            for entity in self.entities:
                if task["domain"] in entity.modules:
                    result = entity.process_task(task["domain"], task["input"])
                    results.append(result)
        # Optimize learning based on raw results
        meta_metric = self.meta_learning.optimize_learning(results)
        print(f"[{self.name}] Meta-metric after optimization: {meta_metric:.2f}")
        return results

    def evolve_system(self):
        """Optimize and restructure the entity system based on meta-learning results."""
        task_results = [30, "Learning the word: collaboration", "def factorial(n): ..."]  # Example data
        meta_metric = self.meta_learning.optimize_learning(task_results)
        print(f"[{self.name}] Meta-metric after optimization: {meta_metric:.2f}")
        self.meta_learning.evolve_entities(self.entities)