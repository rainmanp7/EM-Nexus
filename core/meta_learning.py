# meta_learning.py

class MetaLearning:
    def __init__(self):
        self.task_histories = []

    def optimize_learning(self, task_results):
        """
        Optimize learning pathways based on task results.
        :param task_results: List of task results (can be integers, strings, etc.).
        :return: Average performance metric (e.g., success rate).
        """
        # Calculate a simple performance metric (e.g., success rate)
        success_count = 0
        for result in task_results:
            if result:  # Consider non-zero/non-empty results as successful
                success_count += 1
        success_rate = success_count / len(task_results) if task_results else 0
        return success_rate

    def evolve_entities(self, entities):
        """
        Dynamically restructure entities based on meta-knowledge.
        Only SuperEntities have a learning_engine, so we filter them out.
        """
        print("[MetaLearning] Evolving entity structure...")
        for entity in entities:
            if hasattr(entity, "learning_engine"):  # Check if the entity has a learning_engine
                entity.learning_engine.restructure_model()
            else:
                print(f"[MetaLearning] Skipping evolution for {entity.name} (no learning_engine).")