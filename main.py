from entity_core import SuperEntity
from meta_entity_core import MetaEntity
from holographic_memory import HolographicMemory

def main():
    # Initialize Meta-Entity
    meta_entity = MetaEntity("MetaSuperEntity")

    # Initialize Individual Entities with Holographic Memory and Register Them
    memory_dimensions = 16384
    holographic_memory = HolographicMemory(dimensions=memory_dimensions)

    entity1 = SuperEntity("Entity1", meta_entity, holographic_memory)
    entity2 = SuperEntity("Entity2", meta_entity, holographic_memory)

    meta_entity.register_entity(entity1)
    meta_entity.register_entity(entity2)

    # Example Meta-Task for Collaborative Learning
    meta_task = {
        "description": "Collaborative multi-domain task",
        "sub_tasks": [
            {"domain": "math", "input": {"type": "addition", "a": 10, "b": 20}},
            {"domain": "english", "input": "Learn the word 'superintelligence'"},
            {"domain": "python", "input": "Write a function to calculate factorial"},
        ],
    }

    # Process Meta-Task
    print(f"[Main] Starting Meta-Task: {meta_task['description']}")
    results = meta_entity.process_meta_task(meta_task)
    print(f"[Main] Meta-Task Results: {results}")

    # Store Results into Holographic Memory for Future Retrieval
    for idx, result in enumerate(results):
        task_key = f"Task_{idx + 1}"
        holographic_memory.dynamic_encode(task_key.encode(), str(result).encode())
        print(f"[Main] Stored task result in memory: {task_key} -> {result}")

    # Retrieve Results from Memory for Verification
    for idx in range(len(results)):
        task_key = f"Task_{idx + 1}".encode()
        retrieved_result = holographic_memory.retrieve(task_key)
        print(f"[Main] Retrieved from memory: Task_{idx + 1} -> {retrieved_result[:5]}")

    # Trigger Evolution of the System
    print("[Main] Evolving the system based on task results...")
    meta_entity.evolve_system()

if __name__ == "__main__":
    main()
