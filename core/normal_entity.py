# normal_entity.py

import numpy as np
from core.holographic_memory import HolographicMemory
from core.learning_engine import LearningEngine

class NormalEntity:
    def __init__(self, name, domain, learning_engine=None, memory_store=None):
        """
        Initialize a NormalEntity.
        :param name: Name of the entity.
        :param domain: Domain the entity specializes in (e.g., "math", "english").
        :param learning_engine: Optional LearningEngine for dynamic learning.
        :param memory_store: Optional MemoryStore for persistent knowledge storage.
        """
        self.name = name
        self.domain = domain
        self.learning_engine = learning_engine
        self.memory_store = memory_store
        self.holographic_memory = HolographicMemory(dimensions=16384)  # Local holographic memory

    def process_task(self, task_input):
        """
        Process a task in the entity's domain.
        :param task_input: Input data for the task.
        :return: Dictionary containing the domain and result.
        """
        # Try to retrieve knowledge from holographic memory
        retrieved_result = self.retrieve_knowledge(task_input)
        if retrieved_result:
            print(f"[{self.name}] Retrieved result from memory: {retrieved_result}")
            return {"domain": self.domain, "result": retrieved_result}

        # If no knowledge is found, process the task using domain-specific logic
        result = self._process_task_domain_specific(task_input)

        # Store the result in holographic memory for future use
        self.store_knowledge(task_input, result)
        print(f"[{self.name}] Stored result in memory: {result}")

        # If a learning engine is provided, use it to learn from the task
        if self.learning_engine:
            self.learning_engine.learn(task_input, result)

        return {"domain": self.domain, "result": result}

    def _process_task_domain_specific(self, task_input):
        """
        Process a task using domain-specific logic.
        """
        if self.domain == "math":
            return self.process_math_task(task_input)
        elif self.domain == "english":
            return self.process_english_task(task_input)
        elif self.domain == "science":
            return self.process_science_task(task_input)
        elif self.domain == "python":
            return self.process_python_task(task_input)
        else:
            return "Unsupported domain."

    def process_math_task(self, task_input):
        """
        Process a math task.
        """
        if isinstance(task_input, dict) and "type" in task_input:
            if task_input["type"] == "addition":
                a = task_input.get("a", 0)
                b = task_input.get("b", 0)
                return a + b
        return "Unsupported math task."

    def process_english_task(self, task_input):
        """
        Process an English task.
        """
        if isinstance(task_input, str):
            return f"Learning the word: {task_input}"
        return "Unsupported English task."

    def process_science_task(self, task_input):
        """
        Process a science task.
        """
        if isinstance(task_input, str) and "physics" in task_input:
            return "F = ma (Newton's Second Law)"
        return "Unsupported science task."

    def process_python_task(self, task_input):
        """
        Process a Python task.
        """
        if isinstance(task_input, str) and "function" in task_input:
            if "factorial" in task_input:
                return """def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)"""
        return "Unsupported Python task."

    def store_knowledge(self, input_data, output_data):
        """
        Store knowledge in holographic memory and the database (if available).
        :param input_data: Input data (e.g., task or query).
        :param output_data: Output data (e.g., result or response).
        """
        # Convert input and output to numerical vectors
        input_vector = self._text_to_vector(input_data)
        output_vector = self._text_to_vector(output_data)

        # Store in holographic memory
        self.holographic_memory.dynamic_encode(input_vector, output_vector)

        # Store in the database (if memory_store is provided)
        if self.memory_store:
            self.memory_store.store_knowledge(str(input_data), str(output_data), self.domain)

    def retrieve_knowledge(self, input_data):
        """
        Retrieve knowledge from holographic memory.
        :param input_data: Input data (e.g., task or query).
        :return: Retrieved knowledge as text.
        """
        # Convert input to a numerical vector
        input_vector = self._text_to_vector(input_data)

        # Retrieve from holographic memory
        result_vector = self.holographic_memory.retrieve(input_vector)

        # Decode the vector into a meaningful result
        if self.domain == "math":
            # For math tasks, return the actual solution
            if isinstance(input_data, dict) and "type" in input_data:
                if input_data["type"] == "addition":
                    a = input_data.get("a", 0)
                    b = input_data.get("b", 0)
                    return f"Solution: {a + b}"
        elif self.domain == "english":
            # For English tasks, return the word meaning
            return f"Meaning: {np.mean(result_vector):.2f}"  # Example decoding for English
        elif self.domain == "python":
            # For Python tasks, return the code snippet
            return f"Code: {np.mean(result_vector):.2f}"  # Example decoding for Python
        elif self.domain == "science":
            # For science tasks, return the explanation
            return f"Explanation: {np.mean(result_vector):.2f}"  # Example decoding for science
        else:
            return "Unknown domain."

    @staticmethod
    def _text_to_vector(text, dimensions=1024):
        """
        Convert text into a high-dimensional vector.
        :param text: Input text (can be a string, dictionary, or other types).
        :param dimensions: Number of dimensions for the vector.
        :return: A high-dimensional vector.
        """
        if isinstance(text, dict):
            # Convert dictionary to a string representation
            text = str(sorted(text.items()))  # Sort items to ensure consistent hashing
        elif not isinstance(text, str):
            # Convert other types to string
            text = str(text)
        np.random.seed(hash(text) % (2**32))  # Consistent hash for text
        return np.random.randn(dimensions)