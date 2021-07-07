import random

import Topo
from .Topo import Edge


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

        self.neighbors = {neighbor for neighbor in self.links.otherThan(self)}
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
            # Why don't we append a tuple with the links?
            self.internalLinks.append(Topo.Edge(link1, link2))
        return b

    # Function that creates an edge between two nodes
    def nodeTo(self, n2):
        return Edge(self.n1, n2)

    # Need to figure out the string manipulation part

    # def toString(self):	pass

    # def toFullString(self):	pass
