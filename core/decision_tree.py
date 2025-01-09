import numpy as np

class QuantumDecisionTree:
    def __init__(self):
        self.tree = {}

    def add_decision(self, node, outcomes):
        """Add a decision node with possible outcomes."""
        self.tree[node] = outcomes

    def decide(self, node):
        """Make a quantum-inspired probabilistic decision."""
        if node not in self.tree:
            raise ValueError(f"Node '{node}' not found in decision tree.")
        probabilities = np.array([outcome['probability'] for outcome in self.tree[node]])
        probabilities /= probabilities.sum()  # Normalize probabilities
        choice = np.random.choice(len(self.tree[node]), p=probabilities)
        return self.tree[node][choice]