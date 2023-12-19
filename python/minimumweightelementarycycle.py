import json
from colorama import Fore, Style

class Edge:
    id : int = -1
    node_1_id : int = -1
    node_2_id : int = -1
    weight : int = 1

class Node:
    id : int = -1
    edges : list[Edge] = None  # list of (edge_id, node_id)

class Instance:
    def __init__(self, filepath=None):
        self.nodes : list[Node] = []
        self.edges : list[Edge] = []
        self.maximum_length = 1
        if filepath is not None:
            with open(filepath) as json_file:
                data = json.load(json_file)
                self.maximum_length = data["maximum_length"]
                edges = zip(
                        data["edge_heads"],
                        data["edge_tails"],
                        data["edge_weights"])
                for (node_1_id, node_2_id, weight) in edges:
                    self.add_edge(node_1_id, node_2_id, weight)

    def get_vertices(self) -> set[int]:
        vertices : set[int] = set()
        for edge in self.edges:
            vertices.add(edge.node_1_id)
            vertices.add(edge.node_2_id)
        return vertices

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

    def write(self, filepath):
        data = {"maximum_length": self.maximum_length,
                "edge_heads": [edge.node_1_id for edge in self.edges],
                "edge_tails": [edge.node_2_id for edge in self.edges],
                "edge_weights": [edge.weight for edge in self.edges]}
        with open(filepath, 'w') as json_file:
            json.dump(data, json_file)

    def check(self, filepath):
        print(f"Checker : {filepath}")
        print("-------")
        with open(filepath) as json_file:
            data = json.load(json_file)
            # Compute number of duplicates.
            nodes_in = [0] * len(self.nodes)
            nodes_out = [0] * len(self.nodes)
            for edge_id in data["edges"]:
                edge = self.edges[edge_id]
                nodes_in[edge.node_1_id] += 1
                nodes_out[edge.node_2_id] += 1
            number_of_duplicates = sum(v > 1 for v in nodes_in)
            number_of_duplicates += sum(v > 1 for v in nodes_out)
            # Compute is_connected and is_cycle.
            is_connected = True
            node_id_prec = None
            for edge_id in data["edges"]:
                edge = self.edges[edge_id]
                if node_id_prec is not None:
                    if edge.node_2_id != node_id_prec:
                        is_connected = False
                node_id_prec = edge.node_1_id
            if len(data["edges"]) == 0:
                is_cycle = False
            else:
                is_cycle = (node_id_prec == self.edges[data["edges"][0]].node_2_id)
            
            # Compute lenght.
            length = len(data["edges"])
            # Compute weight.
            weight = sum(self.edges[edge_id].weight
                         for edge_id in data["edges"])

            is_feasible = (
                    (number_of_duplicates == 0)
                    and is_connected
                    and is_cycle
                    and length <= self.maximum_length)
            print(f"Number of duplicates: {number_of_duplicates}")
            print(f"Length: {length}")
            if is_cycle:
                print(f"Is cycle: {Fore.GREEN}{Style.BRIGHT}{is_cycle}{Style.RESET_ALL}")
            else:
                print(f"Is cycle: {Fore.RED}{Style.BRIGHT}{is_cycle}{Style.RESET_ALL}")
            print(f"Is connected: {is_connected}")
            print(f"Feasible: {is_feasible}")
            print(f"Weight: {weight}")
            return (is_feasible, weight)


def bellman_ford(instance : Instance) -> list[list[int]]:
    nb_vertices = len(instance.get_vertices())
    vertices = range(nb_vertices)
    tab_bellmanford : list[list[int]] = [[0 for _ in vertices] for __ in vertices]

    def update_tab(instance : Instance, tab : list[list[int]], i : int, v : int) -> int|None:
        """
            Calcule l_{i+1}(v) = min( l_{i}(v), l_{i}(u) + c_{uv} )
        """
        best_val : int = tab[i][v]
        best_vertex : int|None = None
        for edge in instance.edges:
            if (edge.node_1_id == v):
                u = edge.node_2_id
                if tab[i][u] + edge.weight <= best_val:
                    best_val = tab[i][u] + edge.weight
                    best_vertex = u
        
        tab[i+1][v] = best_val
        return best_vertex

    # Initialisation
    for v in range(1, nb_vertices):
        tab_bellmanford[0][v] = float("inf")
    
    # Boucle principale
    backtrack_dict : dict[tuple[int, int], int|None] = {}
    for i in range(nb_vertices - 1):
        for v in range(nb_vertices):
            backtrack_dict[(i + 1, v)] = update_tab(instance, tab_bellmanford, i, v)
    
    potential_sols : list[list[int]] = []
    for v in vertices:
        visited : list[int] = [v]
        i = nb_vertices - 1
        w = v
        has_been_added = False
        while i > 0:
            u = backtrack_dict[(i, w)]
            if u is None:
                pass

            elif u not in visited:
                # u est un sommet pas encore visité
                visited.append(u)
                w = u
            else:
                # u a déjà été visité par le passé
                u_pos = visited.index(u)
                visited = visited[u_pos:] + [u]
                visited.reverse()
                has_been_added = True
                potential_sols.append(visited)
                w = u
                break
            i -= 1
        
        if not has_been_added and len(visited) > 1:
            can_construct_cycle = False
            u = visited[-1]
            v = visited[0]
            for edge in instance.edges:
                if (edge.node_1_id == u) and (edge.node_2_id == v):
                    can_construct_cycle = True
                    break
            
            if not can_construct_cycle:
                continue
            
            visited.append(visited[0])
            visited.reverse()
            
            total_weight = 0
            for i in range(len(visited) - 1):
                u = visited[i]
                v = visited[i+1]
                for edge in instance.edges:
                    if (edge.node_1_id == v) and (edge.node_2_id == u):
                        total_weight += edge.weight
            if total_weight < 0:
                potential_sols.append(visited)
    if len(potential_sols) == 0:
        return []
    # Keeping only solutions with <= K arcs
    all_vertex_cycles = [cycle for cycle in potential_sols if len(cycle) <= instance.maximum_length+1 and len(cycle) >= 3]
    all_vertex_cycles.sort(key = lambda cycle : len(cycle))

    if len(all_vertex_cycles) == 0:
        return []
    
    all_cycles : list[list[int]] = []
    for cycle_found in all_vertex_cycles:
        solution : list[int] =  []
        for i in range(len(cycle_found) - 1):
            u, v = cycle_found[i], cycle_found[i+1]
            for edge in instance.edges:
                if (edge.node_1_id == v) and (edge.node_2_id == u):
                    solution.append(edge.id)
            i=0
        all_cycles.append(solution)
    return all_cycles

def dynamic_programming(instance : Instance) -> list[int]:
    """
        Takes an instance as an input, returns the list of
        edge_id corresponding to the negative cycle found
    """
    all_cycles = bellman_ford(instance)
    if len(all_cycles) == 0:
        return []
    return all_cycles[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
            "-a", "--algorithm",
            type=str,
            default="dynamic_programming",
            help='')
    parser.add_argument(
            "-i", "--instance",
            type=str,
            help='')
    parser.add_argument(
            "-c", "--certificate",
            type=str,
            default=None,
            help='')

    args = parser.parse_args()

    if args.algorithm == "dynamic_programming":
        instance = Instance(args.instance)
        solution = dynamic_programming(instance)
        if args.certificate is not None:
            data = {"edges": solution}
            with open(args.certificate, 'w') as json_file:
                json.dump(data, json_file)
            print()
            instance.check(args.certificate)
            print(f"{Fore.BLUE}{Style.BRIGHT}{solution}{Style.RESET_ALL}")

    elif args.algorithm == "checker":
        instance = Instance(args.instance)
        instance.check(args.certificate)
