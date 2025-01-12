class Sensorium:
    def __init__(self):
        print("[Sensorium] Initialized.")

    def get_input(self):
        """Simulate input from the environment."""
        return {"type": "stimulus", "data": [1, 0, 1, 0, 1]}

    def perform_action(self, action):
        """Simulate action performance and feedback."""
        reward = action.count(1)  # Reward based on the number of spikes
        result = f"Action {action} performed with reward {reward}."
        return reward, result