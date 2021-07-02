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



class Path:
    """
    A class for all the methods called as Path.<methodname>()

    All methods in this class are static and are solely utilitarian
    """

    @staticmethod
    def edges(path):
        return [Edge(node1, node2) for (node1, node2) in zip(path[:len(path)-1], path[1:])]

    @staticmethod
    def applyCycle():
    # There isn't any call made to this function. If we see a need for this, it can be implemented then.
        pass


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
        self.n = int(lines.pop(0))
        # Why do we have -sys.maxint and not sys.maxint ?
        self.sentinal = Node(self, -1, [-1, -1, -1], sys.maxint)
        self.alpha = float(lines.pop(0))
        self.q = float(lines.pop(0))
        self.k = int(lines.pop(0))
        self.internalLength = math.log(1 / self.q) / self.alpha

        for i in range(self.n):
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


    def getStatistics(self):

        # I don't think we necessarily need to sort each and every one of these lists. Just the min() and the max() will do.
        # We can remove these once we are trying to optimize. For now, I'm doing exactly how the kotlin one does.

        numLinks = [len(node.links) for node in self.nodes].sort()
        numNeighbors = [len([link.theOtherEndOf(node) for link in node.links]) for node in self.nodes].sort()
        linkLengths = [(link.node1.loc - link.node2.loc) for link in self.links].sort()
        linkSuccPossibilities = [pow(math.e, -self.alpha + linkLength) for linkLength in linkLengths].sort()
        numQubits = [node.nQubits for node in self.nodes].sort()

        avgLinks = sum(numLinks) / self.n
        avgNeighbors = sum(numNeighbors) / self.n
        avgLinkLength = sum(linkLengths) / len(self.links)
        avglinkSuccP = sum(linkSuccPossibilities) / len(self.links)
        avgQubits = sum(numQubits) / self.n


        # I didn't understand why the .format() is used here. I've rounded them off to the decimal places provided in the parameter

        return f"""
              
              Topology:
              ${self.n} nodes, ${len(self.links)} links         alpha: ${self.alpha}  q: ${self.q}
              #links     per node                (Max, Avg, Min): ${numLinks[-1]}   	                    ${round(avgLinks, 2)}	    ${numLinks[0]}
              #qubits    per node                (Max, Avg, Min): ${numQubits[-1]}   	                    ${round(avgQubits, 2)}	    ${numQubits[0]}
              #neighbors per node                (Max, Avg, Min): ${numNeighbors[-1]}   	                ${round(avgNeighbors, 2)}	${numNeighbors[0]} 
              length of links       (km)         (Max, Avg, Min): ${round(linkLengths[-1], 2)}	            ${round(avgLinkLength, 2)}	${round(linkLengths[0], 2)} 
              P(entanglement succeed for a link) (Max, Avg, Min): ${round(linkSuccPossibilities[-1], 2)}	${round(avglinkSuccP, 2)}	${round(linkSuccPossibilities[0], 2)}

              """

