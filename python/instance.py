import json

from classes import *
class Edge:
    id = -1
    node_1_id = -1
    node_2_id = -1
    weight = 1


class Node:
    id = -1
    edges = None  # list of (edge_id, node_id)

def build_instance_from_digraph(filepath = None):
    digraph, max_cycle_length = build_from_filename(filepath)
    if digraph is not None:
        instance = Instance()
        instance.maximum_cycle_length = max_cycle_length
        instance.digraph = digraph
        for arc in digraph.arcs:
            instance.add_edge(arc.head, arc.tail, arc.weight)
        return instance
    return None

class Instance:

    def __init__(self, filepath=None):
        self.nodes : list[Node] = []
        self.edges : list[Edge] = []
        self.digraph : Digraph|None = None
        self.visitedList = []
        self.selfless_donors = []
        if filepath is not None:
            self.digraph, self.maximum_cycle_length = build_from_filename(filepath)
            with open(filepath) as json_file:
                data = json.load(json_file)
                self.maximum_cycle_length = data["maximum_cycle_length"]
                self.maximum_path_length = data["maximum_path_length"]
                self.selfless_donors = data["selfless_donors"]
                edges = zip(
                        data["edge_heads"],
                        data["edge_tails"],
                        data["edge_weights"])
                for (node_1_id, node_2_id, weight) in edges:
                    self.add_edge(node_1_id, node_2_id, weight)

    
    def add_node(self):
        node = Node()
        node.id = len(self.nodes)
        node.edges = []
        self.nodes.append(node)

    def add_edge(self, node_id_1, node_id_2, weight):
        edge = Edge()
        edge.id = len(self.edges)
        edge.node_1_id = node_id_1
        edge.node_2_id = node_id_2
        edge.weight = weight
        self.edges.append(edge)
        while max(node_id_1, node_id_2) >= len(self.nodes):
            self.add_node()
        self.nodes[node_id_1].edges.append((edge.id, node_id_2))

    def get_vertices(self) -> set[int]:
        vertices : set[int] = set()
        for edge in self.edges:
            vertices.add(edge.node_1_id)
            vertices.add(edge.node_2_id)
        return vertices

    def write(self, filepath):
        data = {"maximum_cycle_length": self.maximum_cycle_length,
                "maximum_path_length": self.maximum_path_length,
                "selfless_donors": self.selfless_donors,
                "edge_heads": [edge.node_1_id for edge in self.edges],
                "edge_tails": [edge.node_2_id for edge in self.edges],
                "edge_weights": [edge.weight for edge in self.edges]}
        with open(filepath, 'w') as json_file:
            json.dump(data, json_file)

    def check(self, filepath):
        print("Checker")
        print("-------")
        with open(filepath) as json_file:
            data = json.load(json_file)
            # Compute number of duplicates.
            nodes_in = [0] * len(self.nodes)
            nodes_out = [0] * len(self.nodes)
            for edges in data["cycles"] + data["paths"]:
                for edge_id in edges:
                    edge = self.edges[edge_id]
                    nodes_in[edge.node_1_id] += 1
                    nodes_out[edge.node_2_id] += 1
            number_of_duplicates = sum(v > 1 for v in nodes_in)
            number_of_duplicates += sum(v > 1 for v in nodes_out)
            # Compute number_of_wrong_cycles.
            number_of_wrong_cycles = 0
            for edges in data["cycles"]:
                is_connected = True
                node_id_prec = None
                for edge_id in edges:
                    edge = self.edges[edge_id]
                    if node_id_prec is not None:
                        if edge.node_1_id != node_id_prec:
                            is_connected = False
                    node_id_prec = edge.node_2_id
                is_cycle = (node_id_prec == self.edges[edges[0]].node_1_id)
                length = len(edges)
                if (
                        not is_connected
                        or not is_cycle
                        or length > self.maximum_cycle_length):
                    number_of_wrong_cycles += 1
            # Compute number_of_wrong_paths.
            number_of_wrong_paths = 0
            for edges in data["paths"]:
                is_connected = True
                node_id_prec = None
                for edge_id in edges:
                    edge = self.edges[edge_id]
                    if node_id_prec is not None:
                        if edge.node_1_id != node_id_prec:
                            is_connected = False
                    node_id_prec = edge.node_2_id
                length = len(edges)
                if (
                        # not an altruistic donner.
                        not self.edges[edges[0]].node_1_id in self.selfless_donors
                        or not is_connected
                        or length > self.maximum_path_length):
                    number_of_wrong_paths += 1
            # Compute weight.
            weight = sum(self.edges[edge_id].weight
                         for edge_id in edges
                         for edges in data["cycles"] + data["paths"])

            is_feasible = (
                    (number_of_duplicates == 0)
                    and (number_of_wrong_cycles == 0)
                    and (number_of_wrong_paths == 0))
            print(f"Number of duplicates: {number_of_duplicates}")
            print(f"Number of wrong cycles: {number_of_wrong_cycles}")
            print(f"Number of wrong paths: {number_of_wrong_paths}")
            print(f"Feasible: {is_feasible}")
            print(f"Weight: {weight}")
            return (is_feasible, weight)