#!/usr/bin/env python3
"""
    Finding negative cycles of bounded length
    Alternative version of Bellman-Ford
"""
from classes import *
from sys import argv

def bellman_ford(d: Digraph, w : int, k : int):
    """
        Build two dimension recurrence equation for Bellman-Ford,
        rooted at [w], with limited length (length <= k)
    """
    assert w in d.vertices
    
    bellman_tab = [[None for i in range(k + 1)] for v in d.vertices]
    # Initialization
    for v in d.vertices:
        if v == w:
            bellman_tab[v][0] = 0
        else:
            bellman_tab[v][0] = float("inf")
    
    for i in range(k):
        for v in d.vertices:
            if len([bellman_tab[arc.tail][i] + arc.weight for arc in d.arcs if arc.head == v]) == 0:
                bellman_tab[v][i + 1] = float("inf")
            else:
                bellman_tab[v][i + 1] = min([bellman_tab[arc.tail][i] + arc.weight for arc in d.arcs if arc.head == v])
    return bellman_tab

def backtrack_bellman_ford(d: Digraph, tab : list[list[int]], root : int, pos : int):
    # tab[root][pos] is the first negative value found in tab[root]
    # tab[root][0] = 0 by construction of the whole table
    # Objective : finding the sequence of vertices that form a negative cycle, containing root
    
    visited_arcs : list[Arc] = []
    i = pos         # Safe to assume that i >= 2
    v = root
    while i > 0:
        tight_arcs = [arc for arc in d.arcs if arc.head == v and tab[arc.tail][i-1] + arc.weight == tab[arc.head][i]]
        arc = tight_arcs[0]
        i -= 1
        v = arc.tail
        visited_arcs.append(arc)
    visited_arcs.reverse()
    return visited_arcs

def main(filename = None):
    if filename is None:
        return
    # First, we create the digraph, and gather the maxmium length for cycle allowed from the input file
    D, K = build_from_filename(filename)

    # Printing info
    # print(D)

    for w in D.vertices:
        print(f"Bellman ford line obtained, by fixing {chr(65 + w)} as a root : ")
        T = bellman_ford(D, w, K)
        index : int|None = None
        # for line in T:
        #    print(f"\t{line}")
        
        for i, v in enumerate(T[w]): # T[w][i] = v, we check the first i such that v < 0 (if it exists)
            if v < 0:
                index = i
                break
        if index is not None:
            visited_arcs = backtrack_bellman_ford(D, T, w, index)
            for arc in visited_arcs:
                print(f"\t{arc}")
        else:
            print("\tNo cycle has been found.")

if __name__ == "__main__":
    main(argv[-1])