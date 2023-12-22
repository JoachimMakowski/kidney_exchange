import columngenerationsolverpy
import treesearchsolverpy
from copy import deepcopy
from heuristictree import *
from dynamic import *

import json


from instance import *

class PricingSolver:

    def __init__(self, instance):
        self.instance = instance
        # TODO START
        self.vertices_used = None
        # TODO END

    def initialize_pricing(self, columns, fixed_columns):
        # TODO START
        self.vertices_used = [0] * len(self.instance.get_vertices())
        print("\n\ninit pricing")
        for column_id, column_value in fixed_columns:
            column = columns[column_id]
            for row_index, row_coefficient in zip(column.row_indices,
                                                  column.row_coefficients):
                self.vertices_used[row_index] += (column_value*row_coefficient) 
        # TODO END

    def solve_pricing(self, duals):
        # Build subproblem instance.
        # TODO START
        print("\n\nnew dual")
        print(duals)
        print('vertices used in solution', self.vertices_used)
        temp_instance = deepcopy(self.instance)
        for edge in temp_instance.edges:
            edge.weight *= -1
            if self.vertices_used[edge.node_2_id] == 1:
                edge.weight= float("inf")
            else:
                edge.weight+=duals[edge.node_2_id]
        
        temp_instance.digraph, temp_instance.maximum_cycle_length = build_from_instance(temp_instance)
        # TODO END

        # Solve subproblem instance.
        # TODO START
        
        solution = dynamic_programming(temp_instance)
        column = columngenerationsolverpy.Column()
        column.objective_coefficient = 0
        if len(solution) > 0:
            # TODO START
            for edge in solution:
                column.row_indices.append(self.instance.edges[edge].node_2_id)
                column.row_coefficients.append(1)
                column.objective_coefficient+=self.instance.edges[edge].weight
            print('vertices:',[self.instance.edges[edge].node_2_id for edge in solution])
            print('column coef', column.objective_coefficient)
        else:
            '''remeber to add duals[altruistic donor] to the final value of reduced cost'''
            branching_scheme = BranchingScheme(temp_instance, duals)
            output = treesearchsolverpy.iterative_beam_search(
                        branching_scheme,
                        minimum_size_of_the_queue=256,
                        maximum_size_of_the_queue=256)
            solution = branching_scheme.to_solution(output["solution_pool"].best)
            # TODO END

            # TODO START
            if len(solution) > 0:
                column.row_indices.append(self.instance.edges[solution[0]].node_1_id)
                print("vertices visited:",[self.instance.edges[solution[0]].node_1_id]+[self.instance.edges[edge].node_2_id for edge in solution])
                column.row_coefficients.append(1)
                for edge in solution:
                    column.row_indices.append(self.instance.edges[edge].node_2_id)
                    column.row_coefficients.append(1)
                    column.objective_coefficient+=self.instance.edges[edge].weight
            # TODO END

        return [column]


def get_parameters(instance):
    # TODO START
    number_of_constraints = len(instance.get_vertices())
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
        p.row_coefficient_lower_bounds[vertex] = 0
        p.row_coefficient_upper_bounds[vertex] = 1
    # Dummy column objective coefficient.
    p.dummy_column_objective_coefficient = -1
    # TODO END
    # Pricing solver.
    p.pricing_solver = PricingSolver(instance)
    return p


def to_solution(columns, fixed_columns):
    solution = {'cycles': [], 'paths': []}
    path_vs_cycle = 'cycles' #variable to decide whether it's a cycle or a path
    for column, value in fixed_columns:
        # TODO START
        if value > 0:
            s = []
            for index, coef in zip(column.row_indices, column.row_coefficients):
                if coef > 0:
                    s.append(index)
                    if index in instance.selfless_donors:
                        path_vs_cycle = 'paths'
            solution[path_vs_cycle].append(s)
                
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

    elif args.algorithm == "generator":
        import random
        random.seed(0)
        for number_of_nodes in range(101):
            instance = Instance()
            number_of_edges = 6*number_of_nodes if number_of_nodes >= 2 else 0
            number_of_selfless = random.randint(number_of_nodes//16, number_of_nodes//5)
            edges = set()
            for _ in range(number_of_edges):
                node_id_1 = random.randint(0, number_of_nodes - 1)
                node_id_2 = random.randint(0, number_of_nodes - 2)
                while node_id_2 < number_of_selfless:
                    node_id_2 = random.randint(0, number_of_nodes - 2)
                if node_id_2 >= node_id_1:
                    node_id_2 += 1
                edges.add((node_id_1, node_id_2))
            for node_id_1, node_id_2 in edges:
                weight = random.randint(10, 100)
                instance.add_edge(node_id_1, node_id_2, weight)
            instance.maximum_cycle_length = 4
            instance.maximum_path_length = 15
            instance.selfless_donors = [i for i in range(number_of_selfless)]
            instance.write(
                    args.instance + "_" + str(number_of_nodes) + ".json")

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
