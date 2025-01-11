# main.py

from core.entity_core import SuperEntity
from core.meta_entity_core import MetaEntity
from core.holographic_memory import HolographicMemory
from PSDSOm2 import EmergentEntity, Environment
import numpy as np

def main():
    # Initialize MetaEntity and HolographicMemory
    meta_entity = MetaEntity("MetaSuperEntity")
    memory_dimensions = 16384
    holographic_memory = HolographicMemory(dimensions=memory_dimensions)

    # Initialize SuperEntities and register them with the MetaEntity
    entity1 = SuperEntity("Entity1", meta_entity, holographic_memory)
    entity2 = SuperEntity("Entity2", meta_entity, holographic_memory)
    meta_entity.register_entity(entity1)
    meta_entity.register_entity(entity2)

    # Initialize EmergentEntity and Environment
    initial_state = np.random.normal(0, 1, 10)
    possible_actions = np.random.normal(0, 1, 10)
    perception_mechanism = np.random.normal(0, 1, 10)
    emergent_entity = EmergentEntity(initial_state, possible_actions, perception_mechanism)
    environment = Environment({"param1": 1.0, "param2": 2.0})

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
        results = meta_entity.process_meta_task(task)
        print(f"[Main] Meta-Task Results: {results}")

        # Train EmergentEntity using the results
        for result in results:
            emergent_entity.interact_with_environment(environment)
            print(f"[Main] EmergentEntity State: {emergent_entity.state}")

    print("[Main] Evolving the system based on task results...")
    meta_entity.evolve_system()

if __name__ == "__main__":
    main()