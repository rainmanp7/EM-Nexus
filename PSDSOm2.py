import sys
import os
import numpy as np
from entity_controller import EntityController
from memory_store import MemoryStore, HolographicMemory, initialize_database
from core.entity_core import SuperEntity
from core.meta_entity_core import MetaEntity
from core.learning_engine import LearningEngine
from domains.math_module import MathModule
from domains.english_module import EnglishModule
from domains.python_module import PythonModule
from domains.science_module import ScienceModule

# Add the root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

class Environment:
    """Simulates an environment for the emergent entity to interact with."""
    def __init__(self, simulation_parameters):
        self.simulation_parameters = simulation_parameters

    def update(self, state, actions):
        """Simulate the environment's response to the entity's actions."""
        # For simplicity, update the state and perceptions randomly
        new_state = state + np.random.normal(0, 0.1, len(state))
        new_perceptions = np.random.normal(0, 0.1, len(state))
        return new_state, new_perceptions

class EmergentEntity:
    """Represents an emergent entity that interacts with the environment."""
    def __init__(self, state, actions, perceptions):
        self.state = state
        self.actions = actions
        self.perceptions = perceptions
        self.internal_model = None  # Initialize internal model (e.g., neural network)

    def interact_with_environment(self, environment):
        """Simulate interaction with the environment."""
        self.state, self.perceptions = environment.update(self.state, self.actions)

class ExtendedLearningEngine(LearningEngine):
    """Extends the LearningEngine with additional functionality like SOM and PCD."""
    def __init__(self, memory_store):
        super().__init__(memory_store)
        self.persistent_chain = None  # For Persistent Contrastive Divergence (PCD)
        self.som_weights = None  # For Self-Organizing Map (SOM)
        self.input_dim = 10  # Fixed input dimension for SOM
        self.som_map_dim = (5, 5)  # SOM map dimension

    def _ensure_same_length(self, array1, array2):
        """Ensure two arrays have the same length by padding or truncating."""
        len1, len2 = len(array1), len(array2)
        if len1 < len2:
            array1 = np.pad(array1, (0, len2 - len1), mode='constant')
        elif len1 > len2:
            array1 = array1[:len2]
        return array1, array2

    def _to_numerical(self, data):
        """Convert data (string, int, float, dict, etc.) to a numerical array."""
        if isinstance(data, dict):
            return np.array([v for v in data.values() if isinstance(v, (int, float))])
        elif isinstance(data, str):
            return np.array([ord(char) for char in data])
        elif isinstance(data, (int, float)):
            return np.array([data])
        elif isinstance(data, (list, np.ndarray)):
            return np.array(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _preprocess_input(self, input_data):
        """Preprocess input data to match the SOM's expected input dimension."""
        input_vector = self._to_numerical(input_data)
        if len(input_vector) < self.input_dim:
            input_vector = np.pad(input_vector, (0, self.input_dim - len(input_vector)), mode='constant')
        elif len(input_vector) > self.input_dim:
            input_vector = input_vector[:self.input_dim]
        return input_vector

    def initialize_som(self):
        """Initialize the Self-Organizing Map (SOM) weights."""
        self.som_weights = np.random.rand(self.som_map_dim[0], self.som_map_dim[1], self.input_dim)

    def train_som(self, input_data, learning_rate=0.1, radius=1.0, epochs=100):
        """Train the Self-Organizing Map (SOM) with input data."""
        if self.som_weights is None:
            self.initialize_som()  # Initialize SOM weights if not already done

        input_vector = self._preprocess_input(input_data)
        for epoch in range(epochs):
            # Find the Best Matching Unit (BMU)
            bmu_index = np.argmin(np.linalg.norm(self.som_weights - input_vector, axis=2))
            bmu_coords = np.unravel_index(bmu_index, self.som_map_dim)

            # Update weights
            for x in range(self.som_map_dim[0]):
                for y in range(self.som_map_dim[1]):
                    distance = np.linalg.norm(np.array([x, y]) - np.array(bmu_coords))
                    if distance <= radius:
                        self.som_weights[x, y] += learning_rate * (input_vector - self.som_weights[x, y])

    def get_som_output(self, input_data):
        """Get the SOM output for the given input data."""
        input_vector = self._preprocess_input(input_data)
        bmu_index = np.argmin(np.linalg.norm(self.som_weights - input_vector, axis=2))
        return self.som_weights[np.unravel_index(bmu_index, self.som_map_dim)]

    def contrastive_divergence(self, input_data, model_output):
        """Standard Contrastive Divergence."""
        input_vector = self._to_numerical(input_data)
        model_output = self._to_numerical(model_output)
        input_vector, model_output = self._ensure_same_length(input_vector, model_output)

        pos_phase = np.exp(-np.linalg.norm(input_vector - model_output))
        neg_phase = np.exp(-np.linalg.norm(input_vector + np.random.normal(0, 1, len(input_vector))))
        return pos_phase - neg_phase

    def persistent_contrastive_divergence(self, input_data, model_output):
        """Persistent Contrastive Divergence (PCD) with SOM integration."""
        if self.persistent_chain is None:
            self.persistent_chain = np.random.normal(0, 1, len(self._to_numerical(input_data)))

        input_vector = self._to_numerical(input_data)
        model_output = self._to_numerical(model_output)

        # Preprocess both input_vector and model_output to match SOM's input dimension
        input_vector = self._preprocess_input(input_vector)
        model_output = self._preprocess_input(model_output)

        # Use SOM to preprocess the input data
        if self.som_weights is not None:
            input_vector = self.get_som_output(input_vector)

        pos_phase = np.exp(-np.linalg.norm(input_vector - model_output))
        neg_phase = np.exp(-np.linalg.norm(self.persistent_chain))
        self.persistent_chain = model_output  # Update persistent chain
        return pos_phase - neg_phase

    def train_emergent_entity(self, emergent_entity, environment, num_iterations):
        """Train the emergent entity using Contrastive Divergence and SOM."""
        self.initialize_som()  # Ensure SOM is initialized before training
        for _ in range(num_iterations):
            # Emergent behavior simulation
            emergent_entity.interact_with_environment(environment)

            # Contrastive Divergence training
            input_data = np.concatenate([emergent_entity.state, emergent_entity.perceptions])
            output_data = emergent_entity.state  # Use current state as output
            energy = self.contrastive_divergence(input_data, output_data)
            print(f"Energy: {energy}")

            # SOM integration
            self.train_som(input_data)  # Train SOM with input data
            som_output = self.get_som_output(input_data)  # Get SOM output
            # Update Contrastive Divergence with SOM-extracted features
            energy_som = self.persistent_contrastive_divergence(input_data, output_data)
            print(f"Energy (SOM): {energy_som}")

def initialize_system():
    """Initialize the system with entities and learning engine."""
    db_path = "data/test_entity_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    initialize_database(db_path)

    meta_entity = MetaEntity("MetaSuperEntity")
    holographic_memory = HolographicMemory(dimensions=16384)
    learning_engine = ExtendedLearningEngine(MemoryStore(db_path))

    entity1 = SuperEntity("Entity1", meta_entity, holographic_memory)
    entity2 = SuperEntity("Entity2", meta_entity, holographic_memory)
    meta_entity.register_entity(entity1)
    meta_entity.register_entity(entity2)

    # Define initial state, actions, and perceptions for the emergent entity
    initial_state = np.random.normal(0, 1, 10)
    possible_actions = np.random.normal(0, 1, 10)
    perception_mechanism = np.random.normal(0, 1, 10)
    emergent_entity = EmergentEntity(initial_state, possible_actions, perception_mechanism)

    # Define simulation parameters for the environment
    simulation_parameters = {"param1": 1.0, "param2": 2.0}
    environment = Environment(simulation_parameters)

    return meta_entity, entity1, entity2, learning_engine, emergent_entity, environment

def test_learning_methods(meta_entity, entity1, learning_engine, emergent_entity, environment):
    """Test all learning methods with predefined tasks."""
    tasks = [
        {"domain": "math", "input": {"type": "addition", "a": 5, "b": 3}},
        {"domain": "english", "input": "Learn the word 'superintelligence'"},
        {"domain": "python", "input": "Write a function to calculate factorial"},
        {"domain": "science", "input": "Solve a physics problem involving force"},
    ]

    print("Testing Contrastive Divergence...")
    for task in tasks:
        result = entity1.process_task(task["domain"], task["input"])
        energy = learning_engine.contrastive_divergence(task["input"], result)
        print(f"Task: {task['input']}, Energy: {energy}")

    print("\nTesting Emergent Entity Training...")
    learning_engine.train_emergent_entity(emergent_entity, environment, num_iterations=10)

    print("\nTesting Persistent Contrastive Divergence with SOM...")
    for task in tasks:
        result = entity1.process_task(task["domain"], task["input"])
        energy = learning_engine.persistent_contrastive_divergence(task["input"], result)
        print(f"Task: {task['input']}, Energy: {energy}")

def main():
    """Main function to initialize and test the system."""
    meta_entity, entity1, entity2, learning_engine, emergent_entity, environment = initialize_system()
    test_learning_methods(meta_entity, entity1, learning_engine, emergent_entity, environment)
    print("\nAll learning methods tested successfully.")

if __name__ == "__main__":
    main()