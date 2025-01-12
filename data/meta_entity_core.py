# meta_entity_core.py

import sys
import os

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from memory_store import MemoryStore
from core.meta_learning import MetaLearning
from core.entity_core import SuperEntity
from core.holographic_memory import HolographicMemory

class MetaEntity:
    def __init__(self, name):
        """
        Initialize a MetaEntity.
        :param name: Name of the MetaEntity.
        """
        self.name = name
        self.memory = MemoryStore(db_path="data/meta_memory.db")
        self.meta_learning = MetaLearning()
        self.entities = []  # List of SuperEntities managed by the meta-entity
        self.normal_entities = []  # List of NormalEntities managed by the meta-entity
        self.holographic_memory = HolographicMemory(memory_file=f"data/{name}_holographic_memory.npy")

    def register_entity(self, entity):
        """
        Register a SuperEntity with the meta-entity.
        :param entity: The SuperEntity to register.
        """
        self.entities.append(entity)
        print(f"[{self.name}] Registered SuperEntity: {entity.name}")

    def register_normal_entity(self, normal_entity):
        """
        Register a NormalEntity with the meta-entity.
        :param normal_entity: The NormalEntity to register.
        """
        self.normal_entities.append(normal_entity)
        print(f"[{self.name}] Registered NormalEntity: {normal_entity.name}")

    def integrate_task_result(self, entity_name, domain, result):
        """
        Store and analyze task results from an entity.
        :param entity_name: Name of the entity that processed the task.
        :param domain: Domain of the task (e.g., "math", "english").
        :param result: Result of the task.
        """
        self.memory.store_knowledge(f"Result from {entity_name}", str(result), domain)
        print(f"[{self.name}] Integrated result from {entity_name} in domain '{domain}': {result}")

    def process_meta_task(self, meta_task):
        """
        Distribute meta-tasks among entities and coordinate results.
        :param meta_task: A meta-task containing sub-tasks.
        :return: List of results from processing the sub-tasks.
        """
        print(f"[{self.name}] Processing meta-task: {meta_task['description']}")
        results = []
        for task in meta_task["sub_tasks"]:
            # Assign tasks to NormalEntities or SuperEntities based on domain
            if task["domain"] in ["math", "english", "python", "science"]:
                # First, try to assign the task to a NormalEntity with matching domain
                task_assigned = False
                for normal_entity in self.normal_entities:
                    if task["domain"] == normal_entity.domain:
                        result = normal_entity.process_task(task["input"])
                        results.append(result)
                        self.integrate_task_result(normal_entity.name, task["domain"], result["result"])
                        task_assigned = True
                        break

                # If no NormalEntity is available, assign the task to a SuperEntity
                if not task_assigned:
                    for entity in self.entities:
                        result = entity.process_task(task["domain"], task["input"])
                        results.append(result)
                        self.integrate_task_result(entity.name, task["domain"], result)
                        task_assigned = True
                        break

                if not task_assigned:
                    print(f"[{self.name}] No entity available to handle task in domain '{task['domain']}'.")

        # Optimize learning based on raw results
        meta_metric = self.meta_learning.optimize_learning(results)
        print(f"[{self.name}] Meta-metric after optimization: {meta_metric:.2f}")
        return results

    def evolve_system(self):
        """
        Optimize and restructure the entity system based on meta-learning results.
        """
        task_results = [30, "Learning the word: collaboration", "def factorial(n): ..."]  # Example data
        meta_metric = self.meta_learning.optimize_learning(task_results)
        print(f"[{self.name}] Meta-metric after optimization: {meta_metric:.2f}")
        self.meta_learning.evolve_entities(self.entities + self.normal_entities)