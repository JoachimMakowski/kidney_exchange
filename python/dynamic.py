from classes import *
from bellmanford import *
from instance import *

def dynamic_programming(instance : Instance) -> list[int]:
    """
        Takes an instance as an input, returns the list of
        edge_id corresponding to the negative cycle found
    """
    all_cycles = []

    arcs : list[Arc] = []
    for r in instance.digraph.vertices:
        # print(f"Bellman-Ford rooted in {chr(65 + r)}")
        tab = bellman_ford(instance.digraph, r, instance.maximum_cycle_length)
        index : int|None = None
        for i, dist_r in enumerate(tab[r]):
            if dist_r < 0:
                index = i
                break
        
        
        if index is not None:
            arcs = backtrack_bellman_ford(instance.digraph, tab, r, index)
            arcs_id = [a_.arc_id for a_ in arcs]
            
            is_a_permutation = False
            for l in range(len(arcs_id)):
                tmp = arcs_id[l:] + arcs_id[:l]
                if tmp in all_cycles:
                    is_a_permutation = True
                    break
            
            if not is_a_permutation:
                all_cycles.append(arcs_id)

    if len(all_cycles) == 0:
        return []
    return all_cycles[0]