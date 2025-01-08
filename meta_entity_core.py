from memory_store import MemoryStore
from meta_learning import MetaLearning

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
                    results.append(entity.process_task(task["domain"], task["input"]))
        self.meta_learning.optimize_learning(results)
        return results

    def evolve_system(self):
        """Optimize and restructure the entity system based on meta-learning results."""
        task_results = [{"accuracy": 0.9}, {"accuracy": 0.95}, {"accuracy": 0.85}]  # Example data
        meta_metric = self.meta_learning.optimize_learning(task_results)
        print(f"[{self.name}] Meta-metric after optimization: {meta_metric:.2f}")
        self.meta_learning.evolve_entities(self.entities)

if __name__ == "__main__":
    from entity_core import SuperEntity

    meta_entity = MetaEntity("MetaSuperEntity")
    entity1 = SuperEntity("Entity1", meta_entity)
    entity2 = SuperEntity("Entity2", meta_entity)

    meta_entity.register_entity(entity1)
    meta_entity.register_entity(entity2)

    # Example meta-task
    meta_task = {
        "description": "Collaborative task across domains",
        "sub_tasks": [
            {"domain": "math", "input": {"type": "addition", "a": 10, "b": 20}},
            {"domain": "english", "input": "Learn the word 'collaboration'"},
            {"domain": "python", "input": "Write a function to calculate factorial"},
        ],
    }
    results = meta_entity.process_meta_task(meta_task)
    print(f"[{meta_entity.name}] Meta-task results: {results}")
