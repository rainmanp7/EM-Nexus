# main.py

from core.entity_core import SuperEntity
from core.meta_entity_core import MetaEntity
from core.holographic_memory import HolographicMemory
from core.normal_entity import NormalEntity
from core.learning_engine import LearningEngine
from memory_store import MemoryStore
import numpy as np

def main():
    # Initialize HolographicMemory
    memory_dimensions = 16384
    holographic_memory = HolographicMemory(dimensions=memory_dimensions)

    # Initialize 2 MetaEntities
    meta_entity1 = MetaEntity("MetaEntity1")
    meta_entity2 = MetaEntity("MetaEntity2")

    # Initialize 2 SuperEntities and register them with MetaEntities
    super_entity1 = SuperEntity("SuperEntity1", meta_entity1, holographic_memory)
    super_entity2 = SuperEntity("SuperEntity2", meta_entity2, holographic_memory)
    meta_entity1.register_entity(super_entity1)
    meta_entity2.register_entity(super_entity2)

    # Initialize 2 NormalEntities with LearningEngine and MemoryStore
    learning_engine = LearningEngine(MemoryStore("data/entity_memory.db"))
    normal_entity1 = NormalEntity("NormalEntity1", domain="math", learning_engine=learning_engine)
    normal_entity2 = NormalEntity("NormalEntity2", domain="english", learning_engine=learning_engine)
    meta_entity1.register_normal_entity(normal_entity1)
    meta_entity2.register_normal_entity(normal_entity2)

    # Define collaborative tasks
    collaborative_tasks = [
        {
            "description": "Collaborative multi-domain task",
            "sub_tasks": [
                {"domain": "math", "input": {"type": "addition", "a": 10, "b": 20}},
                {"domain": "english", "input": "Learn the word 'superintelligence'"},
                {"domain": "python", "input": "Write a function to calculate factorial"},
            ],
        },
    ]

    # Collaborative training loop
    for task in collaborative_tasks:
        print(f"[Main] Starting Meta-Task: {task['description']}")
        results = meta_entity1.process_meta_task(task)
        print(f"[Main] Meta-Task Results: {results}")

        # Train entities using the results
        for result in results:
            if isinstance(result, dict) and "domain" in result:
                if result["domain"] == "math":
                    result_value = float(result["result"]) if isinstance(result["result"], (int, float)) else 0
                    input_data = np.array([task["sub_tasks"][0]["input"]["a"], task["sub_tasks"][0]["input"]["b"]], dtype=float)
                    normal_entity1.learning_engine.learn(input_data, result_value)
                elif result["domain"] == "english":
                    result_value = float(hash(str(result["result"])))
                    input_data = np.array([float(hash(task["sub_tasks"][1]["input"]))], dtype=float)
                    normal_entity2.learning_engine.learn(input_data, result_value)
                elif result["domain"] == "python":
                    result_value = float(hash(str(result["result"])))
                    input_data = np.array([float(hash(task["sub_tasks"][2]["input"]))], dtype=float)
                    super_entity1.learning_engine.learn(input_data, result_value)

        print(f"[Main] Entities trained using task results.")

    print("[Main] Evolving the system based on task results...")
    meta_entity1.evolve_system()

if __name__ == "__main__":
    main()