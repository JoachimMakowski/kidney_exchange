from functools import total_ordering

class BranchingScheme:


    @total_ordering
    class Node:

        id = None
        father = None
        # TODO START
        number_of_nodes_visited = None
        visited = None
        j = None
        length = None
        # TODO END
        guide = None
        next_child_pos = 0

        def __lt__(self, other):
            if self.guide != other.guide:
                return self.guide < other.guide
            return self.id < other.id

    def __init__(self, instance, donor_weights):
        self.instance = instance
        self.donor_weights = donor_weights
        self.id = 0

    def root(self):
        node = self.Node()
        node.father = None
        # TODO START
        node.number_of_nodes_visited=0
        node.visited = 0
        node.j = 0
        node.length = 0
        # TODO END
        node.guide = 0
        node.id = self.id
        self.id += 1
        return node

    def next_child(self, father):
        # TODO START
        if father.number_of_nodes_visited:
            if len(self.instance.nodes[father.j].edges) == 0:
                return None
            j_next = self.instance.nodes[father.j].edges[father.next_child_pos][1]
            
            if (father.visited >> j_next) & 1:
                father.next_child_pos +=1
                return None
            if father.number_of_nodes_visited + 1 > self.instance.maximum_path_length:
                father.next_child_pos +=1
                return None
            child = self.Node()
            child.father = father
            child.visited = father.visited + (1 << j_next)
            child.number_of_nodes_visited = father.number_of_nodes_visited + 1
            child.j = j_next
            child.length = father.length + self.instance.edges[self.instance.nodes[father.j].edges[father.next_child_pos][0]].weight
            child.next_child_pos = 0
            child.guide = child.length
            child.id = self.id
            self.id += 1
            father.next_child_pos +=1
            return child            
        else:
            if not len(self.instance.selfless_donors):
                return None
            j_next = self.instance.selfless_donors[father.next_child_pos]
            #if j_next > max(max(self.instance.edge_heads),max(self.instance.edge_tails)):
                #return None
            child = self.Node()
            child.father = father
            child.visited = father.visited + (1 << j_next)
            child.number_of_nodes_visited = father.number_of_nodes_visited + 1
            child.j = j_next
            child.length = father.length + self.donor_weights[j_next]
            child.next_child_pos = 0
            child.guide = child.length
            child.id = self.id
            self.id += 1
            father.next_child_pos +=1
            return child  
        # TODO END

    def infertile(self, node):
        # TODO START
        if node.number_of_nodes_visited:
            res = node.next_child_pos == len(self.instance.nodes[node.j].edges)
        else:
            res = node.next_child_pos == len(self.instance.selfless_donors)
        return res
        # TODO END

    def leaf(self, node):
        # TODO START
        if node is None or node.number_of_nodes_visited == self.instance.maximum_path_length:
            return True
        return False
        # TODO END

    def bound(self, node_1, node_2):
        # TODO START
        return False
        # TODO END

    # Solution pool.

    def better(self, node_1, node_2):
        # TODO START
        # Compute the objective value of node_1.
        d1 = node_1.length
        # Compute the objective value of node_2.
        d2 = node_2.length
        return d1 < d2
        # TODO END

    def equals(self, node_1, node_2):
        # TODO START
        return False
        # TODO END

    # Dominances.

    def comparable(self, node):
        # TODO START
        return True
        # TODO END

    class Bucket:

        def __init__(self, node):
            self.node = node

        def __hash__(self):
            # TODO START
            return hash((self.node.j, self.node.visited))
            # TODO END

        def __eq__(self, other):
            # TODO START
            return (
                    # Same last location.
                    self.node.j == other.node.j
                    # Same visited locations.
                    and self.node.visited == other.node.visited)
            # TODO END

    def dominates(self, node_1, node_2):
        # TODO START
        #if node_1.length >= node_2.length:
        #    return True
        return False
        # TODO END

    # Outputs.

    def display(self, node):
        # TODO START
        # Compute the objective value of node.
        d = node.length
        return str(d)
        # TODO END

    def to_solution(self, node):
        # TODO START
        locations = []
        node_tmp = node
        while node_tmp.father is not None:
            #self.instance.edges[self.instance.nodes[node_tmp.father.j].edges[node_tmp.father.next_child_pos]]
            for i in self.instance.nodes[node_tmp.father.j].edges:
                if i[1] == node_tmp.j:
                    locations.append(i[0])
            
            node_tmp = node_tmp.father
        locations.reverse()
        print(locations)
        return locations