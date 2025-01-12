import networkx as nx

class EntanglementHub:
    def __init__(self, entity_name):
        self.entity_name = entity_name
        self.graph = nx.DiGraph()

    def synchronize_states(self, entity, state_data):
        print(f"[{self.entity_name}] Synchronizing state with {entity}...")
        self.graph.add_node(self.entity_name, state=state_data)
        self.graph.add_edge(self.entity_name, entity, relation="shared_state")