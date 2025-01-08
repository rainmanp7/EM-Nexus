import numpy as np
from holographic_memory import HolographicMemory

class MathModule:
    def __init__(self, memory_dimensions=16384):
        self.memory = HolographicMemory(dimensions=memory_dimensions)

    def store_math_problem(self, problem, solution):
        """
        Encode a math problem and its solution into holographic memory.
        :param problem: String representation of the problem.
        :param solution: Numerical solution to the problem.
        """
        problem_vector = np.random.randn(self.memory.dimensions)
        solution_vector = np.array([solution] * self.memory.dimensions)
        self.memory.dynamic_encode(problem_vector, solution_vector)

    def retrieve_solution(self, problem):
        """
        Retrieve a solution for a given math problem from holographic memory.
        :param problem: String representation of the problem.
        :return: Retrieved solution (approximation).
        """
        problem_vector = np.random.randn(self.memory.dimensions)
        return self.memory.retrieve(problem_vector)

# Example usage
if __name__ == "__main__":
    math_module = MathModule()
    math_module.store_math_problem("5 + 5", 10)
    retrieved_solution = math_module.retrieve_solution("5 + 5")
    print(f"Retrieved solution: {retrieved_solution[:5]}")
