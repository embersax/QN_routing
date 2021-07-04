import math
from .Node import Node
from .Link import Link
from .Node import Node
from .Link import Link
from ..utils.CollectionUtils import PriorityQueue
import random
import hashlib
import sys
import re

hopLimit = 15

"""This class defines an edge of the graph"""


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


"""This class represents the topology of the Quantum Network"""


def priority(n1, n2):
    if n1.id < n2.id:
        return -1
    elif n1.id > n2.id:
        return 1
    else:
        return

class Path:
    """
    A class for all the methods called as Path.<methodname>()

    All methods in this class are static and are solely utilitarian
    """

    @staticmethod
    def edges(path):
        return [Edge(node1, node2) for (node1, node2) in zip(path[:len(path) - 1], path[1:])]

    @staticmethod
    def applyCycle():
        # There isn't any call made to this function. If we see a need for this, it can be implemented then.
        pass



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
            linkLine=lines.pop(0)
            tmp=list(map(lambda x: int(x) ,line.split(" ")))
            tmp1=tmp[0:2]
            tmp2=list(map(lambda x: self.nodes[x],tmp1))
            n1,n2=sorted(tmp2, key=lambda x: x.id, reverse=True)
            nm=tmp1[2]

            if n1.id >=n2.id:
                sys.exit()
            for i in range(1,nm):
                link=Link(self,n1,n2,+(n1.loc - n2.loc))
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

    # This is the implementation of the shortestPath algorithm
    # We first need to take all edges (in both directions)
    def shortestPath(self, edges, src, dst, fStateMetric):
        reversedEdges = [Edge(edge.n1, edge.otherThan(edge.n1)) for edge in edges]
        allNodes = [edge.toList() for edge in reversedEdges]
        # I'm having a lot of trouble implementing the neighborsOf part of the code-
        neighborsOf = {}
        prevFromSrc = {}
        D = dict.fromkeys(self.nodes, float('inf'))
        # Do we need to have a comparator? Can we just use a def
        q = PriorityQueue()
        D[src.id] = 0.0
        q.push(src.nodeTo(self.sentinal), priority(src, self.sentinal))
        while q:
            # Not sure about this pop, but I think it is correct based on the comparator
            (w, prev) = q.pop()
            if w in prevFromSrc: continue
            prevFromSrc[w] = prev

            if w == dst:
                path = []
                cur = dst
                while cur != self.sentinal:
                    path.insert(0, cur)
                    cur = prevFromSrc[cur]
                return D[dst.id], path

            for p in neighborsOf[w]:
                neighbor, weight = p[0], p[1]
                newDist = D[w.id] + weight
                oldDist = D[neighbor.id]
                if oldDist > newDist:
                    D[neighbor.id] = newDist
                    q.push(neighbor.nodeTo(w), priority(neighbor, w))
        # I don't understand this return statement.
        return float('inf'), []


    # Returns all routes for two nodes
    def getAllRoutes(self, n1_, n2_):
        n1, n2 = [min(n1_, n2_), max(n1_, n2_)]
        linksAndEdges = {}
        for link in self.links:
            linksAndEdges[link] = link.node1.nodeTo(link.node2)
        topoStr = '\n'.join(str(el) for el in set(linksAndEdges))
        digest = hashlib.sha256()
        digest.update(bytearray(topoStr))
        # I'm not sure what to do with the routeStorage part
        result = []
        range_ = self.kHopNeighbors(self.nodes[n1], (hopLimit + 1) / 2) + self.kHopNeighbors(self.nodes[n2], (hopLimit + 1) / 2)
        # If result is empty
        if not result:
            # Find all
            def dfs(l, remainingNeighbors):
                if l[-1] == n2:
                    result.append(list(l))
                elif len(l) > hopLimit:
                    return
                else:
                    filtered = filter(lambda it: it in range_, self.nodes[l[-1]].neighbors)
                    map_ = {}
                    for node in filtered:
                        map_[node] = node.id
                    for _, v in map_:
                        if v not in l and remainingNeighbors:
                            l.append(v)
                            dfs(l, list(set(remainingNeighbors) - set(self.nodes[v])))
                            l.pop(len(l)-1)
            dfs(list(n1), list(set(self.nodes[n2].neighbors) - set(self.nodes[n1])))

            # Sort via Dijkstra
            # I'm not sure about this part
            p = {}
            return result


    # Returns all routes between two nodes
    def getAllElementCycles(self):
        linksAndEdges = {}
        for link in self.links:
            linksAndEdges[link] = link.node1.nodeTo(link.node2)
        topoStr = '\n'.join(str(el) for el in set(linksAndEdges))
        digest = hashlib.sha256()
        digest.update(bytearray(topoStr))
        # I'm not sure what to do with the routeStorage part
        result = []
        # If result is empty
        if not result:
            resultSet = []
            for n in self.nodes:
                # Not sure but maybe it has to be +1 to be inclusive
                for length in range(3,10+1):
                    def dfs(l):
                        tmp = l[-1].neighbors.intersection(l)
                        if len(l) == length:
                            if tmp == set(l[l.size - 2], n):
                                tmp1 = {}
                                for node in l: tmp1[node] = node.id
                                l = tmp1
                                m = l.index(min(l))
                                resultSet.append(l[m:len(l)] + l[0:m])
                        elif len(tmp) <= 1:
                            filtered = filter(lambda it: len(l) == 1 or it != l[len(l)-2], l[-1].neighbors)
                            for f in filtered:
                                l.append(f)
                                dfs(l)
                                l.pop(len(l) - 1)
                    dfs(list(n))
            result.extend(resultSet)
            self.saveRoutes()
        # I don't understand this return statement
        return dict.fromkeys(result, self.nodes)

    def getStatistics(self):

        # I don't think we necessarily need to sort each and every one of these lists. Just the min() and the max() will do.
        # We can remove these once we are trying to optimize. For now, I'm doing exactly how the kotlin one does.

        numLinks = [len(node.links) for node in self.nodes].sort()
        numNeighbors = [len([link.otherThan(node) for link in node.links]) for node in self.nodes].sort()
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

    def kHopNeighbors(self, root, k):
        if k > self.k:  return set(self.nodes)

        registered = [False for _ in self.nodes]
        stack = []
        stack.append(root)
        registered[root.id] = True

        def work():
            current = stack[-1]

            if len(stack) <= k + 1:
                unregisteredNeighbors = set(filter(lambda x: x.id not in registered,
                                                   [Edge(link.node1, link.node2).otherThan(current) for link in
                                                    current.links]))

                for unregisteredNeighbor in unregisteredNeighbors:
                    registered[unregisteredNeighbor.id] = True
                    stack.append(unregisteredNeighbor)
                    work()
                    stack.pop()

        work()

        res = []
        for idx, val in enumerate(registered):
            if val:
                res.append(self.nodes[idx])
            else:
                res.append(self.sentinal)

        res = set(filter(lambda x: x != self.sentianl, res))

        return res

    def kHopNeighborLinks(self, root, k):
        registered = [False for _ in self.nodes]
        result = set()

        stack = []
        stack.append(root)
        registered[root.id] = True

        def work():
            current = stack[-1]
            result.union(self.current.links)

            if len(stack) <= k + 1:
                unregisteredNeighbors = set(filter(lambda x: x.id not in registered,
                                                   [Edge(link.node1, link.node2).otherThan(current) for link in
                                                    current.links]))

                for unregisteredNeighbor in unregisteredNeighbors:
                    registered[unregisteredNeighbor.id] = True
                    stack.append(unregisteredNeighbor)
                    work()
                    stack.pop()

        work()

        return result

    def getEstablishedEntanglements(self, node1, node2):
        stack = []
        stack.append((None, node1))

        result = []

        while stack:
            incoming, current = stack.pop()

            if current == node2:
                path = [node2]
                inc = incoming

                while inc.node1 != node1 and inc.node2 != node2:
                    prev = inc.node2 if inc.node1 == path[-1] else inc.node1
                    inc = [link.otherThan(inc) for link in prev.internalLinks if inc in link]
                    path.append(prev)

                path.append(node1)
                result.append(list(reversed(path)))
                continue

            outgoingLinks = list(filter(lambda link: link.entangled and not link.swappedAt(current),
                                        current.links)) if incoming == None else [link.otherThan(incoming) for link in
                                                                                  list(filter(lambda
                                                                                                  internalLink: incoming in internalLink,
                                                                                              current.internalLinks))]

            for outgoingLink in outgoingLinks:
                stack.append([outgoingLink, Edge(outgoingLink.node1, outgoingLink.node2).otherThan(current)])

        return result

    def isClean(self):
        areLinksClean, areNodesClean = True, True
        for link in self.links:
            areLinksClean = areLinksClean and not link.entangled and not link.assigned and link.notSwapped()
        for node in self.nodes:
            areNodesClean = areNodesClean and not node.internalLinks and node.nQubits == node.remainingQubits

        return areLinksClean and areNodesClean

    # This also has not been called anywhere
    def linksBetween(self, node1, node2):
        return list(filter(lambda link: node2 == link.node1 or node2 == link.node2, [link for link in node1.links]))









