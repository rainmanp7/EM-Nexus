from math_module import MathModule
from english_module import EnglishModule
from programming_module import ProgrammingModule

class EntityController:
    def __init__(self):
        self.math_module = MathModule()
        self.english_module = EnglishModule()
        self.programming_module = ProgrammingModule()

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
