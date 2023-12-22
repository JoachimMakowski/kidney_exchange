"""
    Useful classes for solving problems
"""
import json

class Arc:
    def __init__(self, arc_id: int, tail : int, head : int, weight : int):
        self.arc_id : int = arc_id
        self.head   : int = head
        self.tail   : int = tail
        self.weight : int = weight
    
    def __str__(self):
        return f"@{self.arc_id} : {chr(65 + self.tail)} --[{self.weight}]--> {chr(65 + self.head)}"

class Digraph:
    def __init__(self):
        self.vertices   : set[int] = set()
        self.arcs       : list[Arc] = []
    
    def add_arc(self, tail : int, head : int, weight : int):
        self.vertices.add(tail)
        self.vertices.add(head)
        self.arcs.append(Arc(len(self.arcs), tail, head, weight))
    
    def __str__(self) -> str:
        s = f"V = {self.vertices}\n"
        for arc in self.arcs:
            s = s + "\t" + arc.__str__() + "\n"
        return s
    
def build_from_filename(filename = None):
    if filename is None:
        return None, None
    
    d = Digraph()
    m = 0 # Max cycle length
    with open(filename) as json_file:
        data = json.load(json_file)
        m = data["maximum_cycle_length"]
        arcs = zip(data["edge_heads"], data["edge_tails"], data["edge_weights"])
        for (v, u, w_uv) in arcs:
            d.add_arc(u, v, w_uv)
    return (d, m)

def build_from_instance(instance = None):
    if instance is None:
        return None, None
    
    d = Digraph()
    m = 0 # Max cycle length
    
    m = instance.maximum_cycle_length
    for edge in instance.edges:
        d.add_arc(edge.node_2_id, edge.node_1_id, edge.weight)
    return (d, m)
