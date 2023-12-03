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
        self.maximum_length = 1
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

    #TODO START
    def add_edge_2(self, edge):
        self.edges.append(edge)
        while max(edge.node_id_1, edge.node_id_2) >= len(self.nodes):
            self.add_node()
        self.nodes[edge.node_id_1].edges.append((edge.id, edge.node_id_2))
    #TODO END

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

def dynamic_programming(instance : Instance) -> list[int]:
    """
        Takes an instance as an input, returns 
    """
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
            
    # print("Bellman-Ford table obtained with dynamic programming :")
    # for line_no, line in enumerate(tab_bellmanford):
    #     print(f"Line no. {line_no} : {line}")
    
    # print("Backtrack dictionnary :")
    # for i in range(1, nb_vertices):
    #     best_choice_in_line = [chr(65 + backtrack_dict[(i, v)]) if backtrack_dict[(i, v)] is not None else "∅" for v in vertices]
    #     print(f"Line no.{i} :", best_choice_in_line)
    
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
                if (edge.node_1_id == v) and (edge.node_2_id == u):
                    can_construct_cycle = True
            
            if can_construct_cycle:
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
                
            
            # sum(instance.edges[edge_id].weight for edge_id in data["edges"])
    
    # Keeping only solutions with <= K arcs
    potential_sols = [cycle for cycle in potential_sols if len(cycle) < instance.maximum_cycle_length + 1]
    potential_sols.sort(key = lambda cycle : len(cycle))

    if potential_sols == []:
        return []
    
    neg_cycle = tuple(potential_sols[0])
    solution : list[int] = []
    for i in range(len(neg_cycle) - 1):
        u, v = neg_cycle[i], neg_cycle[i+1]
        for edge in instance.edges:
            if (edge.node_1_id == v) and (edge.node_2_id == u):
                solution.append(edge.id)
    return solution

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
        instance_to_dyn_prog = Instance()
        for vertex in range(len(instance.get_vertices())):
            profit = duals[vertex]
            
            if profit <= 0:
                continue
            '''temp_edge = Edge()
            temp_edge.id = instance.edges[edge].id
            temp_edge.node_1_id = instance.edges[edge].node_1_id
            temp_edge.node_2_id = instance.edges[edge].node_2_id
            temp_edge.weight = -1*profit
            instance_to_dyn_prog.add_edge_2(edge)'''

        # TODO END

        # Solve subproblem instance.
        # TODO START
        solution = dynamic_programming(instance_to_dyn_prog)
        # TODO END

        # Retrieve column.
        column = columngenerationsolverpy.Column()
        # TODO START
        column.objective_coefficient = 0
        
        arcs = [0] * len(self.instance.edges)
        for j in solution:
            #if self.instance.edges[j] > 0:
            column.row_indices.append(self.instance.edges[j].node_1_id)
            column.row_coefficients.append(1)
            column.objective_coefficient += self.instance.edges[j].weight
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
    for vertex in range(number_of_constraints):
        p.row_lower_bounds[vertex] = 0
        p.row_upper_bounds[vertex] = 1
        #coefficients
        p.row_coefficient_lower_bounds[vertex] = 0
        p.row_coefficient_upper_bounds[vertex] = 1

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
