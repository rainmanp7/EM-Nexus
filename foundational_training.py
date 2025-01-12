# foundational_training.py

from core.entity_core import SuperEntity
from core.meta_entity_core import MetaEntity
from core.holographic_memory import HolographicMemory
from core.normal_entity import NormalEntity
from core.learning_engine import LearningEngine
from memory_store import MemoryStore
import numpy as np

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

    print("[Training] Foundational knowledge loaded successfully!")

def load_foundational_knowledge(math_entity, english_entity, programming_entity, science_entity):
    print("[Training] Loading foundational knowledge...")

    # Math examples
    math_examples = [
        ("Count to 5", [1, 2, 3, 4, 5]),
        ("Add 2 and 3", 5),
    ]
    for problem, solution in math_examples:
        math_entity.store_knowledge(problem, solution)

    # English examples
    english_examples = [
        ("Spell 'cat'", "c-a-t"),
        ("Form a sentence with 'cat'", "The cat is sleeping."),
    ]
    for prompt, result in english_examples:
        english_entity.store_knowledge(prompt, result)

    # Programming examples
    programming_examples = [
        ("Print 'Hello, World!'", "print('Hello, World!')"),
        ("Create a loop to count to 5", "for i in range(1, 6): print(i)"),
    ]
    for task, code in programming_examples:
        programming_entity.store_knowledge(task, code)

    # Science examples
    science_examples = [
        ("What is Newton's Second Law?", "F = ma (Force equals mass times acceleration)."),
        ("What is water's chemical formula?", "H2O."),
    ]
    for question, answer in science_examples:
        science_entity.store_knowledge(question, answer)

if __name__ == "__main__":
    main()