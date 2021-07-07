import random


# from Quantum.Topo import to

# from topo.Topo import to
class Edge:
    def __init__(self, n1, n2):
        self.p = (n1, n2)
        self.n1 = n1
        self.n2 = n2

    # Converts the tuple that conforms an edge into a list.
    def toList(self):
        return list(self.p)

    # Given a node n, returns the other node in the edge.
    def otherThan(self, n):
        if n == self.n1:
            return self.n2
        elif n == self.n2:
            return self.n1
        else:
            raise RuntimeError("Neither")

    # Returns true if the node n is either n1 or n2.
    def contains(self, n):
        return self.n1 == n or self.n2 == n

    # The hashcode of the edge is the xor function between the ids of both nodes.
    def hashCode(self):
        return self.n1.id ^ self.n2.id

    # An exact same edge shares both n1 and n2. Note that the edge is bidirectional.
    def equals(self, other):
        return (type(other) is Edge) and (self.p == other.p or reversed(self.p) == other.p)


# Creates an edge between two nodes
def to(node1, node2):
    return Edge(node1, node2)


class Node:
    # The constructor of a node. Parameters:
    # A topology topo
    # An id
    # A localization of the node in the network
    # The number of qubits, n_qubits, assigned to this node
    def __init__(self,
                 topo,
                 id,
                 loc,
                 nQubits):

        self.topo = topo
        self.id = id
        self.loc = loc
        self.nQubits = nQubits

        self.links = []
        self.internalLinks = []

        # self.neighbors = {neighbor for neighbor in self.links.theOtherEndOf(self)}

        self.remainingQubits = nQubits

    # add property to neighbors can avoid the error on init part when links is empty list, but I havn't done the by
    # lazy part
    @property
    def neighbors(self):
        return {neighbor for neighbor in self.links.theOtherEndOf(self)}

    # Attempts link-swapping inside a node
    def attemptSwapping(self, link1, link2):
        if link1.node1 == self:
            assert not link1.swap1
            link1.swap1 = True
        else:
            assert not link1.swap2
            link1.swap2 = True

        if link2.node1 == self:
            assert not link2.swap1
            link2.swap1 = True
        else:
            assert not link2.swap2
            link2.swap2 = True

        b = random.uniform(0, 1) <= self.topo.q

        if b:
            self.internalLinks.append(to(link1, link2))
        return b

    # Need to figure out the string manipulation part

    # def toString(self):	pass

    # def toFullString(self):	pass
