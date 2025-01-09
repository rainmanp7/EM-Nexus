# domains/python_module.py

class PythonModule:
    def __init__(self, learning_engine):
        self.engine = learning_engine

    def process(self, task_input):
        """
        Process a Python-related task.
        """
        if isinstance(task_input, str) and "function" in task_input:
            return self.create_function(task_input)
        return "Unsupported Python task."

    def create_function(self, task):
        """
        Generate a Python function based on the task.
        """
        if "factorial" in task:
            return """def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)"""
        return "Function generation not supported for this task."