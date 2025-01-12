# cross_training.py

from core.entity_core import SuperEntity
from core.meta_entity_core import MetaEntity
from core.holographic_memory import HolographicMemory
from core.normal_entity import NormalEntity
from core.learning_engine import LearningEngine
from memory_store import MemoryStore
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Initialize the system
    holographic_memory = HolographicMemory(dimensions=16384)
    meta_entity = MetaEntity("MetaEntity1")
    learning_engine = LearningEngine(MemoryStore("data/entity_memory.db"))
    
    # Initialize entities
    math_entity = NormalEntity("MathEntity", domain="math", learning_engine=learning_engine)
    english_entity = NormalEntity("EnglishEntity", domain="english", learning_engine=learning_engine)
    programming_entity = NormalEntity("ProgrammingEntity", domain="python", learning_engine=learning_engine)
    science_entity = NormalEntity("ScienceEntity", domain="science", learning_engine=learning_engine)
    
    # Register entities with the meta-entity
    meta_entity.register_normal_entity(math_entity)
    meta_entity.register_normal_entity(english_entity)
    meta_entity.register_normal_entity(programming_entity)
    meta_entity.register_normal_entity(science_entity)

    # Load foundational knowledge
    load_foundational_knowledge(math_entity, english_entity, programming_entity, science_entity)

    # Cross-train the system
    total_connections = cross_train_system(math_entity, english_entity, programming_entity, science_entity)

    # Print the final result with separate lines for "complete" and "total connections"
    print("\n[Training] Cross-training complete!")
    print("Total connections made: {}\n".format(total_connections))

def load_foundational_knowledge(math_entity, english_entity, programming_entity, science_entity):
    print("[Training] Loading foundational knowledge...")

    # Math examples (updated to use dictionary format)
    math_examples = [
        ({"type": "count", "values": [1, 2, 3, 4, 5]}, [1, 2, 3, 4, 5]),  # Dictionary format for counting
        ({"type": "addition", "a": 2, "b": 3}, 5),  # Dictionary format for addition
    ]
    for problem, solution in math_examples:
        math_entity.store_knowledge(problem, solution)

    # English examples (unchanged)
    english_examples = [
        ("Spell 'cat'", "c-a-t"),
        ("Form a sentence with 'cat'", "The cat is sleeping."),
    ]
    for prompt, result in english_examples:
        english_entity.store_knowledge(prompt, result)

    # Programming examples (unchanged)
    programming_examples = [
        ("Print 'Hello, World!'", "print('Hello, World!')"),
        ("Create a loop to count to 5", "for i in range(1, 6): print(i)"),
    ]
    for task, code in programming_examples:
        programming_entity.store_knowledge(task, code)

    # Science examples
    science_examples = [
        ("What is the chemical formula for water?", "H2O"),
        ("Describe the process of photosynthesis", "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."),
    ]
    for question, answer in science_examples:
        science_entity.store_knowledge(question, answer)

def cross_train_system(math_entity, english_entity, programming_entity, science_entity):
    print("[Training] Cross-training the system...")
    total_connections = 0

    # Example 1: Transfer counting (Math) to loops (Programming)
    math_problem = {"type": "count", "values": [1, 2, 3, 4, 5]}  # Updated to dictionary format
    math_solution = math_entity.retrieve_knowledge(math_problem)
    programming_task = "Create a loop to count to 5"
    programming_solution = programming_entity.retrieve_knowledge(programming_task)
    print(f"[Cross-Training] Math: {math_problem} -> {math_solution}")
    print(f"[Cross-Training] Programming: {programming_task} -> {programming_solution}")
    print("Connection: Counting in math is similar to looping in programming.")
    total_connections += 1

    # Example 2: Transfer sentence structure (English) to code structure (Programming)
    english_prompt = "Form a sentence with 'cat'"
    english_solution = english_entity.retrieve_knowledge(english_prompt)
    programming_task = "Print 'Hello, World!'"
    programming_solution = programming_entity.retrieve_knowledge(programming_task)
    print(f"[Cross-Training] English: {english_prompt} -> {english_solution}")
    print(f"[Cross-Training] Programming: {programming_task} -> {programming_solution}")
    print("Connection: Sentence structure in English is similar to code structure in programming.")
    total_connections += 1

    # Example 3: Transfer addition (Math) to summing a list (Programming)
    math_problem = {"type": "addition", "a": 2, "b": 3}  # Updated to dictionary format
    math_solution = math_entity.retrieve_knowledge(math_problem)
    programming_task = "Sum the list [2, 3]"
    programming_solution = "sum([2, 3])"
    print(f"[Cross-Training] Math: {math_problem} -> {math_solution}")
    print(f"[Cross-Training] Programming: {programming_task} -> {programming_solution}")
    print("Connection: Addition in math is similar to summing a list in programming.")
    total_connections += 1

    # Example 4: Transfer chemical formulas (Science) to variable naming (Programming)
    science_question = "What is the chemical formula for water?"
    science_solution = science_entity.retrieve_knowledge(science_question)
    programming_task = "Define a variable for water"
    programming_solution = "water = 'H2O'"
    print(f"[Cross-Training] Science: {science_question} -> {science_solution}")
    print(f"[Cross-Training] Programming: {programming_task} -> {programming_solution}")
    print("Connection: Chemical formulas in science can be represented as variables in programming.")
    total_connections += 1

    # Example 5: Transfer photosynthesis (Science) to process automation (Programming)
    science_question = "Describe the process of photosynthesis"
    science_solution = science_entity.retrieve_knowledge(science_question)
    programming_task = "Automate a process to simulate photosynthesis"
    programming_solution = "def photosynthesis(light, water, co2): return 'glucose'"
    print(f"[Cross-Training] Science: {science_question} -> {science_solution}")
    print(f"[Cross-Training] Programming: {programming_task} -> {programming_solution}")
    print("Connection: Biological processes like photosynthesis can be simulated through automation in programming.")
    total_connections += 1

    return total_connections

if __name__ == "__main__":
    main()