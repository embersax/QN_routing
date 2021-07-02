import math
from .Node import Node
from .Link import Link
import random
import sys
import re

"""This class defines an edge of the graph"""


class Edge:
    def __init__(self, n1, n2):
        self.p = (n1, n2)

    # Converts the tuple that conforms an edge into a list.
    def toList(self):
        return list(self.p)

    # Given a node n, returns the other node in the edge.
    def otherThan(self, n):
        if n == self.p[0]:
            return self.p[1]
        elif n == self.p[1]:
            return self.p[0]
        else:
            raise RuntimeError("Neither")

    # Returns true if the node n is either n1 or n2.
    def contains(self, n):
        return self.p[0] == n or self.p[1] == n

    # The hashcode of the edge is the xor function between the ids of both nodes.
    def hashCode(self):
        return self.p[0].id ^ self.p[1].id

    # An exact same edge shares both n1 and n2. Note that the edge is bidirectional.
    def equals(self, other):
        return (type(other) is Edge) and (self.p == other.p or reversed(self.p) == other.p)


"""This class represents the topology of the Quantum Network"""


class Topo:
    def __init__(self, input):
        self.n = 0
        self.alpha = 0
        self.q = 0
        self.k = 0
        self.nodes = []
        self.links = []
        self.sentinal = None
        self.nodeDigits = 0
        self.linkDigits = 0
        self.distanceDigits = 0
        self.internalLength = 0

        lines = list(filter(lambda x: x != "", re.sub("""\s*//.*$""", "", input.splitlines())))
        n = int(lines.pop(0))
        # Why do we have -sys.maxint and not sys.maxint ?
        self.sentinal = Node(self, -1, [-1, -1, -1], sys.maxint)
        self.alpha = float(lines.pop(0))
        self.q = float(lines.pop(0))
        self.k = int(lines.pop(0))
        self.internalLength = math.log(1 / self.q) / self.alpha

        for i in range(n):
            line = lines.pop(0)
            # I think we need it to be float(x)
            tmp = list(map(lambda x: float(x), line.split(" ")))
            node = Node(self, i, [tmp[1], tmp[2]], int(tmp[0]))
            self.nodes.append(node)

        self.nodeDigits = round(math.ceil(math.log10(float(len(self.nodes)))))

        while len(lines):
            linkLine = lines.pop(0)
            # I think it should be linkLine.split(" ") instead of line.split(" ")
            tmp = list(map(lambda x: int(x), linkLine.split(" ")))
            tmp1 = tmp[0:2]
            tmp2 = list(map(lambda x: self.nodes[x], tmp1))
            # I made a tuple with n1 and n2, as Shouqian did.
            (n1, n2) = sorted(tmp2, key=lambda x: x.id, reverse=True)
            # I think it has to be tmp, as tmp1[2] is None.
            nm = tmp[2]
            assert n1.id < n2.id
            # I'm not sure but maybe as range is not inclusive for the upper limit, it has to be range(1,nm+1),
            # according to Shouqian's code.
            for i in range(1, nm + 1):
                link = Link(self, n1, n2, +(n1.loc - n2.loc))
                self.links.append(link)
                n1.links.append(link)
                n2.links.append(link)

        self.linkDigits = round(math.ceil(math.log10(len(self.links))))
        flat_map = lambda f, xs: (y for ys in xs for y in f(ys))
        self.distanceDigits = round(math.ceil(math.log10(
            max(list(map(lambda x: x.l, self.links)) +
                list(map(lambda x: math.abs(x), list(flat_map(lambda x: x.loc, self.nodes))))))))

    # All links, entanglements and swappings are set to false.
    def clearEntanglements(self):
        for link in self.links:
            link.clearEntanglement()
            link.swap1, link.swap2 = False, False

        for node in self.nodes:
            node.internalLinks.clear()
            assert node.nQubits == node.remainingQubits

    # We only clear the swappings in phase 4, but not the entanglements.
    def clearOnlyPhase4(self):
        for link in self.links:
            link.swap1, link.swap2 = False, False

        for node in self.nodes:
            node.internalLinks.clear()

