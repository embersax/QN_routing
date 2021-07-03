import random

from .Topo import Edge


class Node:
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

        self.neighbors = {neighbor for neighbor in self.links.theOtherEndOf(self)}
        self.remainingQubits = nQubits

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
            # Why don't we append a tuple with the links?
            self.internalLinks.append((link1, link2))
        return b

    # Function that creates an edge between two nodes
    def nodeTo(self, n2):
        return Edge(self.n1, n2)
    
    # Need to figure out the string manipulation part
    
    # def toString(self):	pass

    # def toFullString(self):	pass
