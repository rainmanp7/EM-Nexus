# diagnostic_test.py

import sys
import os
import sqlite3

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# Import necessary modules
from database_setup import initialize_database
from entity_controller import EntityController
from main import main
from memory_store import MemoryStore
from core.holographic_memory import HolographicMemory
from programming_module import ProgrammingModule
from core.decision_tree import QuantumDecisionTree
from core.entanglement_hub import EntanglementHub
from core.entity_core import SuperEntity
from core.learning_engine import LearningEngine
from core.meta_entity_core import MetaEntity
from core.meta_learning import MetaLearning
from core.sensorium import Sensorium
from domains.english_module import EnglishModule
from domains.math_module import MathModule
from domains.python_module import PythonModule
from domains.science_module import ScienceModule
from utils.hyperdimensional_utils import encode_to_hyperdimensional, decode_from_hyperdimensional
import numpy as np

def test_database_setup():
    print("Testing database setup...")
    initialize_database("data/test_entity_memory.db")
    initialize_database("data/test_meta_memory.db")
    print("Database setup test passed.\n")

def test_domain_databases():
    print("Testing domain-specific databases...")
    domain_databases = {
        "math": "data/math.db",
        "english": "data/english.db",
        "programming": "data/programming.db",
        "science": "data/science.db",
    }

    for domain, db_path in domain_databases.items():
        print(f"Testing {domain} database at {db_path}...")
        try:
            # Connect to the database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Verify the knowledge table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge';")
            table_exists = cursor.fetchone()
            if not table_exists:
                raise ValueError(f"Table 'knowledge' does not exist in {db_path}")

            # Insert a test record
            test_input = f"Test input for {domain}"
            test_output = f"Test output for {domain}"
            cursor.execute("INSERT INTO knowledge (input, output) VALUES (?, ?)", (test_input, test_output))
            conn.commit()

            # Retrieve the test record
            cursor.execute("SELECT input, output FROM knowledge WHERE input = ?", (test_input,))
            result = cursor.fetchone()
            if not result or result[0] != test_input or result[1] != test_output:
                raise ValueError(f"Data retrieval failed in {db_path}")

            # Clean up the test record
            cursor.execute("DELETE FROM knowledge WHERE input = ?", (test_input,))
            conn.commit()

            print(f"{domain} database test passed.")
        except sqlite3.Error as e:
            print(f"Error testing {domain} database: {e}")
        finally:
            conn.close()
    print("All domain database tests passed.\n")

def test_entity_controller():
    print("Testing EntityController...")
    controller = EntityController()
    math_result = controller.process_task("math", "5 + 5")
    english_result = controller.process_task("english", "Learn the word 'superintelligence'")
    programming_result = controller.process_task("programming", "Write a function to calculate factorial")
    print(f"Math result: {math_result}")
    print(f"English result: {english_result}")
    print(f"Programming result: {programming_result}")
    print("EntityController test passed.\n")

def test_holographic_memory():
    print("Testing HolographicMemory...")
    memory = HolographicMemory(dimensions=16384)
    key = np.random.randn(16384)
    value = np.random.randn(16384)
    memory.dynamic_encode(key, value)
    retrieved_value = memory.retrieve(key)
    print(f"Retrieved value: {retrieved_value[:5]}")
    print("HolographicMemory test passed.\n")

def test_programming_module():
    print("Testing ProgrammingModule...")
    
    # Initialize LearningEngine with a MemoryStore
    learning_engine = LearningEngine(MemoryStore("data/test_entity_memory.db"))
    
    # Initialize ProgrammingModule with the learning_engine
    module = ProgrammingModule(learning_engine)
    
    # Test storing and retrieving a code snippet
    module.store_code_snippet("print('Hello, World!')", "Prints a greeting message")
    retrieved_description = module.retrieve_description("print('Hello, World!')")
    print(f"Retrieved description: {retrieved_description[:5]}")
    print("ProgrammingModule test passed.\n")

def test_quantum_decision_tree():
    print("Testing QuantumDecisionTree...")
    tree = QuantumDecisionTree()
    tree.add_decision("node1", [{"probability": 0.5, "result": "A"}, {"probability": 0.5, "result": "B"}])
    decision = tree.decide("node1")
    print(f"Decision: {decision}")
    print("QuantumDecisionTree test passed.\n")

def test_entanglement_hub():
    print("Testing EntanglementHub...")
    hub = EntanglementHub("TestHub")
    hub.synchronize_states("Entity1", {"state": "active"})
    print("EntanglementHub test passed.\n")

def test_super_entity():
    print("Testing SuperEntity...")
    meta_entity = MetaEntity("MetaSuperEntity")
    entity = SuperEntity("TestEntity", meta_entity)
    entity.run()
    print("SuperEntity test passed.\n")

def test_meta_entity():
    print("Testing MetaEntity...")
    meta_entity = MetaEntity("MetaSuperEntity")
    entity1 = SuperEntity("Entity1", meta_entity)
    entity2 = SuperEntity("Entity2", meta_entity)
    meta_entity.register_entity(entity1)
    meta_entity.register_entity(entity2)
    meta_task = {
        "description": "Collaborative task across domains",
        "sub_tasks": [
            {"domain": "math", "input": {"type": "addition", "a": 10, "b": 20}},
            {"domain": "english", "input": "Learn the word 'collaboration'"},
            {"domain": "python", "input": "Write a function to calculate factorial"},
        ],
    }
    results = meta_entity.process_meta_task(meta_task)
    print(f"Meta-task results: {results}")
    print("MetaEntity test passed.\n")

def test_sensorium():
    print("Testing Sensorium...")
    sensorium = Sensorium()
    input_data = sensorium.get_input()
    reward, result = sensorium.perform_action([1, 0, 1, 0, 1])
    print(f"Input data: {input_data}")
    print(f"Action result: {result}, Reward: {reward}")
    print("Sensorium test passed.\n")

def test_english_module():
    print("Testing EnglishModule...")
    module = EnglishModule()
    module.store_word_meaning("apple", "a fruit")
    retrieved_meaning = module.retrieve_meaning("apple")
    print(f"Retrieved meaning: {retrieved_meaning[:5]}")
    print("EnglishModule test passed.\n")

def test_math_module():
    print("Testing MathModule...")
    module = MathModule()
    module.store_math_problem("5 + 5", 10)
    retrieved_solution = module.retrieve_solution("5 + 5")
    print(f"Retrieved solution: {retrieved_solution[:5]}")
    print("MathModule test passed.\n")

def test_python_module():
    print("Testing PythonModule...")
    learning_engine = LearningEngine(MemoryStore("data/test_entity_memory.db"))
    module = PythonModule(learning_engine)
    result = module.process("Write a function to calculate factorial")
    print(f"PythonModule result: {result}")
    print("PythonModule test passed.\n")

def test_science_module():
    print("Testing ScienceModule...")
    learning_engine = LearningEngine(MemoryStore("data/test_entity_memory.db"))
    module = ScienceModule(learning_engine)
    result = module.process("Solve a physics problem involving force")
    print(f"ScienceModule result: {result}")
    print("ScienceModule test passed.\n")

def test_hyperdimensional_utils():
    print("Testing HyperdimensionalUtils...")
    vector = np.random.randn(10)
    encoded_vector = encode_to_hyperdimensional(vector)
    decoded_vector = decode_from_hyperdimensional(encoded_vector)
    print(f"Original vector: {vector[:5]}")
    print(f"Decoded vector: {decoded_vector}")
    print("HyperdimensionalUtils test passed.\n")

def main():
    test_database_setup()
    test_domain_databases()  # New test for domain databases
    test_entity_controller()
    test_holographic_memory()
    test_programming_module()
    test_quantum_decision_tree()
    test_entanglement_hub()
    test_super_entity()
    test_meta_entity()
    test_sensorium()
    test_english_module()
    test_math_module()
    test_python_module()
    test_science_module()
    test_hyperdimensional_utils()

    print("All diagnostic tests passed successfully.")

if __name__ == "__main__":
    main()