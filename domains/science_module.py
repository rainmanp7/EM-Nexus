# domains/science_module.py

class ScienceModule:
    def __init__(self, learning_engine):
        self.engine = learning_engine

    def process(self, task_input):
        """
        Process a science-related task.
        """
        if isinstance(task_input, str) and "physics" in task_input:
            return self.solve_physics_problem(task_input)
        return "Unsupported science task."

    def solve_physics_problem(self, task):
        """
        Solve a physics problem.
        """
        if "force" in task:
            return "F = ma (Newton's Second Law)"
        return "Physics problem not supported."