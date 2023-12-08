import columngenerationsolverpy

import json


class Edge:
    id = -1
    node_1_id = -1
    node_2_id = -1
    weight = 1


class Node:
    id = -1
    edges = None  # list of (edge_id, node_id)


class Instance:

    def __init__(self, filepath=None):
        self.nodes = []
        self.edges = []
        self.cycleListe = []
        self.M=[]
        self.maximum_length = 1
        filepath = "data/kidneyexchange/instance_6.json"
        if filepath is not None:
            with open(filepath) as json_file:
                data = json.load(json_file)
                self.maximum_length = data["maximum_cycle_length"]
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
    
    #TODO Le j° vertex appartient au i°cycle
    def matrice(self,CycleList):
        m = [[0 for _ in range(len(CycleList))] for _ in range(len(self.get_vertices()))]
        for cycle_no, cycle in enumerate(CycleList):
            for edge_id in cycle:
                u = self.edges[edge_id].node_2_id
                v = self.edges[edge_id].node_1_id
                m[u][cycle_no] = 1
                m[v][cycle_no] = 1
        return m
    #TODO END
    
    def depthFirst(self, graph, currentVertex, visited):
        if len(visited)<=14:
            visited.append(currentVertex)
            for vertex in graph[currentVertex]:
                if vertex not in visited:
                    self.depthFirst(graph, vertex, visited.copy())
        self.visitedList.append(visited)

    
    def combPath(self):
        dict_neighbors = {} # 
        edge: Edge
        for edge in self.edges:
            if edge.node_1_id not in dict_neighbors.keys():
                dict_neighbors[edge.node_1_id] = [edge.node_2_id]
            else:
                dict_neighbors[edge.node_1_id].append(edge.node_2_id)
        print(dict_neighbors)
        for donneur in self.selfless_donors:
            #self.recursif([donneur],adj,listPath)
            self.depthFirst(dict_neighbors, donneur, [])

        return self.visitedList

    def write(self, filepath):
        data = {"maximum_cycle_length": self.maximum_cycle_length,
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
            for edges in data["cycles"]:
                for edge_id in edges:
                    edge = self.edges[edge_id]
                    nodes_in[edge.node_1_id] += 1
                    nodes_out[edge.node_2_id] += 1
            number_of_duplicates = sum(v > 1 for v in nodes_in)
            number_of_duplicates += sum(v > 1 for v in nodes_out)
            # Compute number_of_wrong_paths.
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
            # Compute weight.
            weight = sum(self.edges[edge_id].weight
                         for edge_id in edges
                         for edges in data["cycles"])

            is_feasible = (
                    (number_of_duplicates == 0)
                    and (number_of_wrong_cycles == 0))
            print(f"Number of duplicates: {number_of_duplicates}")
            print(f"Number of wrong cycles: {number_of_wrong_cycles}")
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

class PricingSolver:

    def __init__(self, instance):
        self.instance = instance
        # TODO START
        self.vertices_used = None
        # TODO END

    def initialize_pricing(self, columns, fixed_columns):
        # TODO START
        self.vertices_used = [0] * len(self.instance.get_vertices())
        for column_id, column_value in fixed_columns:
            column = columns[column_id]
            for row_index, row_coefficient in zip(column.row_indices,
                                                  column.row_coefficients):
                self.vertices_used[row_index] += (column_value*row_coefficient)            
        # TODO END

    def solve_pricing(self, duals):
        # Build subproblem instance.
        # TODO START
        cout_max=-1
        cycle_max = 0
        for j, cycle in enumerate(instance.cycleListe):
            
            profit = sum([edge.weight for edge in self.instance.edges if edge.id in cycle])
            for vertex in range(len(instance.get_vertices())):
                profit -= duals[vertex]*self.instance.M[vertex][j]
            
            if profit <= 0:
                continue
            if profit > cout_max:
                cout_max=profit
                cycle_max=cycle
        if cout_max == -1:
            return []
        

        # TODO END

        # Retrieve column.
        column = columngenerationsolverpy.Column()
        # TODO START
        column.objective_coefficient = 0
        
        
        for edge in self.instance.edges:
            if edge.id in cycle_max:
            #if self.instance.edges[j] > 0:
                column.row_indices.append(edge.node_1_id)
                column.row_coefficients.append(1)
                column.objective_coefficient += edge.weight
        # TODO END
        return [column]


def get_parameters(instance : Instance):
    # TODO START
    number_of_constraints = len(instance.get_vertices()) #number of vertices
    p = columngenerationsolverpy.Parameters(number_of_constraints)
    # Objective sense.
    p.objective_sense = "max"
    # Column bounds.
    p.column_lower_bound = 0
    p.column_upper_bound = 1
    # Row bounds.
    for edge in instance.edges:
        edge.weight *= -1

    cycleList : list[list[int]] = bellman_ford(instance)
    instance.cycleListe =cycleList
    m = instance.matrice(cycleList)
    instance.M=m    
    
    for edge in instance.edges:
        edge.weight *= -1

    for vertex in range(number_of_constraints):
        p.row_lower_bounds[vertex] = 0
        p.row_upper_bounds[vertex] = 1
        #coefficients
        p.row_coefficient_lower_bounds[vertex] = min(m[vertex])
        p.row_coefficient_upper_bounds[vertex] = max(m[vertex])

    # Dummy column objective coefficient.
    p.dummy_column_objective_coefficient = -1

    # TODO END
    # Pricing solver.
    p.pricing_solver = PricingSolver(instance)
    return p


def to_solution(columns, fixed_columns):
    solution = {'cycles': []}
    for column, value in fixed_columns:
        # TODO START
        if value > 0:
            s = []
            for index, coef in zip(column.row_indices, column.row_coefficients):
                if coef > 0:
                    s.append(index)
            solution["cycles"].append(s)
        # TODO END
    return solution


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
            "-a", "--algorithm",
            type=str,
            default="column_generation",
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

    if args.algorithm == "checker":
        instance = Instance(args.instance)
        instance.check(args.certificate)

    elif args.algorithm == "column_generation":
        instance = Instance(args.instance)
        output = columngenerationsolverpy.column_generation(
                get_parameters(instance))

    else:
        instance = Instance(args.instance)
        parameters = get_parameters(instance)
        if args.algorithm == "greedy":
            output = columngenerationsolverpy.greedy(
                    parameters)
        elif args.algorithm == "limited_discrepancy_search":
            output = columngenerationsolverpy.limited_discrepancy_search(
                    parameters)
        solution = to_solution(parameters.columns, output["solution"])
        if args.certificate is not None:
            with open(args.certificate, 'w') as json_file:
                json.dump(solution, json_file)
            print()
            instance.check(args.certificate)