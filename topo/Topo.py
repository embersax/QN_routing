import math
import random
import sys
import re
from itertools import combinations, groupby

from topo.Node import Node
from topo.Link import Link
from utils.Disjoinset import Disjointset
from utils.utils import *
import numpy as np
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
def to(node1,node2):
    return Edge(node1,node2)

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
        x=input.splitlines()
        x_next=list(map( lambda y:  re.sub("""\s*//.*$""", "" ,y)      ,x))
        lines = list(filter(lambda x: x != "", x_next))
        n = int(lines.pop(0))
        # Why do we have -sys.maxint and not sys.maxint ?
        ## Python 3 has no maxint. Replacing it with maxsize
        self.sentinal = Node(self, -1, [-1, -1, -1], sys.maxsize)
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
            (n1, n2) = sorted(tmp2, key=lambda x: x.id)
            # I think it has to be tmp, as tmp1[2] is None.
            nm = tmp[2]
            assert n1.id < n2.id
            # I'm not sure but maybe as range is not inclusive for the upper limit, it has to be range(1,nm+1),
            # according to Shouqian's code.
            for i in range(1, nm + 1):
                link = Link(self, n1, n2, length(list_minus(n1.loc,n2.loc)))
                self.links.append(link)
                n1.links.append(link)
                n2.links.append(link)

        self.linkDigits = round(math.ceil(math.log10(len(self.links))))
        flat_map = lambda f, xs: (y for ys in xs for y in f(ys))

        ## abs is defined in the standard library itself. Replacing math.abs with abs
        self.distanceDigits = round(math.ceil(math.log10(
            max(list(map(lambda x: x.l, self.links)) +
                list(map(lambda x: abs(x), list(flat_map(lambda x: x.loc, self.nodes))))))))

    @staticmethod
    def listtoString(s, seperator):
        s = list(map(lambda x:str(x), s))
        return seperator.join(s)

    def toString(self):
        return self.toConfig()

    @staticmethod
    def groupby_dict(s, f):
        tmp = {}
        for a, b in groupby(s, f):
            for thing in b:
                tmp[a] = b
        return tmp

    def toConfig(self):
        nl = "\n"
        tmp_dict = self.grouby_dict(self.links, lambda x: x.n1.to(x.n2))
        return f"""
            {self.n}
            {self.alpha}
            {self.q}
            {self.k}
            {self.listtoString(list(map(lambda node: str(node.nQubits) + self.listtoString(node.loc, " "), self.nodes)), nl)}
            {self.listtoString(list(map(lambda x: str(x.n1.id) + str(len(tmp_dict[x])), tmp_dict)), nl)}
            """

    def toFullString(self):
        nl = "\n"
        tmp_dict = self.grouby_dict(self.links, lambda x: x.n1.to(x.n2))
        return f"""
            {self.n}
            {self.alpha}
            {self.q}
            {self.k}
            {self.listtoString(list(map(lambda node: str(node.remainingQubits) + "/" + str(node.nQubits) + self.listtoString(node.loc, " "), self.nodes)), nl)}
            {self.listtoString(list(map(lambda x: str(x.n1.id) + str(len(tmp_dict[x])) + str(len(tmp_dict[x])), tmp_dict)), nl)}
            """

    def widthPhase2(self, path):
        tmp1 = min(list(map(lambda x: x.remainingQubits / 2, path[1:-1])))
        p1 = tmp1 if tmp1 else sys.maxint
        p2 = min(list(
            map(lambda x: sum(1 for link in x[0].links if ((link.n1 == x[1] | link.n2 == x[1]) & (not link.assigned))),
                list(zip(path[:-1], path[1:])))))
        return min(path[0].remainingQubits, path[-1].remainingQubits, p1, p2)

    def widthPhase4(self, path):

        return min(list(map(lambda x: sum(
            1 for link in x[0].links if ((link.n1 == x[1] | link.n2 == x[1]) & link.entangled & link.notSwapped())),
                            list(zip(path[:-1], path[1:])))))

    def e(self, path, mul, oldP):
        s = len(path) - 1
        P = [0.0 for _ in range(mul + 1)]
        p = [0.0] + list(map(lambda x: math.exp(-self.alpha * +(x.n1.loc - x.n2.loc)), path.edges()))
        start = s

        ## Not sure what 'm', 'C_' are. Worth checking again.

        if sum(oldP) == 0:
            for i in range(0, mul):
                oldP[i] = C_(mul, i) * p[1] ** (m) + (1 - p[1]) ** (mul - m)
            start = 2
        assert len(oldP) == mul + 1

        for k in range(start, s):
            for i in range(0, mul):
                exactlyM = C_(mul, i) * p[k] ** (i) * (1 - p[k]) ** (mul - i)
                atLeastM = sum(list(map(lambda x: C_(mul, x) * p[k] ** (x) * (1 - p[k]) ** (mul - x),
                                        [l for l in range(i + 1, mul)]))) + exactlyM
                tmp = 0
                for l in range(i + 1, mul):
                    tmp += oldP[l]
                P[i] = oldP[i] * atLeastM + exactlyM * tmp
            for i in range(0, mul):
                oldP[i] = P[i]

        assert abs(sum(oldP) - 1.0) < 0.0001

        return sum(list(map(lambda x: x * oldP[x], [i for i in range(1, mul)]))) * self.q ** (s - 1)


## Using the debug mode on Shoqian's and Ruilin's code line by line and assessing their equality and making changes accordingly.

def generate(n, q, k, a, degree):
    alpha = a

    ## Shoqian didn't explicitly create it in Topo.kt but it was assigned the value 100 in configs.kt
    ## Manually assigning the value. But, at a later point in time, have to move this to configs.py file
    edgeLen = 100.0
    controllingD = math.sqrt(edgeLen * edgeLen / n)
    links = []
    # #Added nodeLocs
    nodeLocs = []
    while len(nodeLocs) < n:
        ## element is a list but x is not. The '-' operator is not defined. Please recheck this part.
        # "+" operator matches the unaryPlus operator of Shouqian's code of Double Array
        element = [random.uniform(0, 1) * 100 for _ in range(2)]
        if all(length(list_minus(x,element)) > controllingD / 1.2 for x in nodeLocs):
            nodeLocs.append(element)
    nodeLocs = sorted(nodeLocs, key=lambda x: x[0] + int(x[1] * 10 / edgeLen) * 1000000)

    def argument_function(beta):
        links.clear()
        ## An efficient algorithm is already available in itertools. Replacing with combinations() call
        tmp_list =list( combinations([i for i in range(0, n)],2))
        for i in range(len(tmp_list)):
            (l1, l2) = list(map(lambda x: nodeLocs[x], [tmp_list[i][0], tmp_list[i][1]]))
            d = length(list_minus(l2,l1))
            if d < 2 * controllingD:
                l = min([random.uniform(0, 1) for i in range(1,51)])
                r = math.exp(-beta * d)
                if l < r:
                    # to functions needed to be implemented
                    links.append( [tmp_list[i][0],tmp_list[i][1]])
        tmp1=len(links)

        return 2 * float(len(links)) / n

    # Can't fully debug unless the dynSearch and disjointSet are actually implemented
    # dynSearch needed to be implememnted,I just realized beta is not used Shouqian's code
    beta = dynSearch(0.0, 20.0, float(degree),argument_function, False, 0.2)
    # DisjoinSet needed to be implemmented
    disjoinSet = Disjointset(n)
    for i in range(len(links)):
        disjoinSet.merge(links[i][0], links[i][1])
    # compute  ccs: =(o unitil n).map.map.groupby.map.sorted
    ### finish debugging before this part
    t1= list(map( lambda x: [x,disjoinSet.getRepresentative(x)] , [i for i in range(0,n)] ))
    # from shouqian's code it seems that t1 and t2 have the same transformation and t2 is basically same as t1, I'm not sure
    t2 = list( map( lambda x:[x[0],x[1]] ,t1 ))
    # [x,disjoinSet.getgetRepresentative(x)]
    t3=groupby_dict(t2,lambda x: x[1])
    t4=list(map(lambda x: list(map(lambda x:x[0] ,t3[x]) ) ,t3))
    ccs =sorted(t4,key=lambda x:-len(x))

    biggest = ccs[0]
    for i in range(1, len(ccs)):
        # this part I'm not sure since shuffle changes the order of original ccs list
        ## ccs1 is not defined. Please check
        tmp1 = random.shuffle(ccs[i])[0:3]
        for j in range(len(tmp1)):
            nearest = sorted(biggest, key=lambda x: sum(nodeLocs[x] - nodeLocs[tmp1[j]]))[0]
            tmp2 = sorted([nearest, tmp1[j]])
            links.append(to(tmp2[0],tmp2[1]))

    ## groupby_dict, listtoString are functions Topo class objects. They seem to be used outside the class. So, making them static in this case.
    flat_map = lambda f, xs: (y for ys in xs for y in f(ys))
    #     retrive the flatten list first
    tmp_list = list(flat_map(lambda x: [x[0],x[1]], links))
    # retriev dictionary to iterate
    tmp_dict =groupby_dict_(tmp_list, lambda x: x)

    for key in tmp_dict:
        if len(tmp_dict[key]) / 2 < 5:
            # not sure if takes the right index, needed to double check
            nearest = sorted([i for i in range(0, n)], key=lambda x: length(    list_minus(nodeLocs[x],nodeLocs[key])))[
                      1:int(6 - len(tmp_dict[key]) / 2)]

            tmp_list = list(map(lambda x: [sorted([x, key])[0], sorted([x, key])[1]],
                                nearest))
            # need to check what is added to links here
            for item in tmp_list:
                links.append(item)

    nl = "\n"
    tmp_string1 = Topo.listtoString(
        list(map(lambda x: f"{int(random.uniform(0, 1) * 5 + 10)} " + Topo.listtoString(x, " "), nodeLocs)), nl)
    tmp_string2 = Topo.listtoString(
        list(map(lambda x: f"{x[0]} " + f"{x[1] } " + f"{int(random.uniform(0, 1) * 5 + 3)}",
                 links))
        , nl)
    return Topo(f"""{n}
{alpha}
{q}
{k}
{tmp_string1}
{tmp_string2}
"""

                )

### I added Vamsi's updated code
    def getStatistics(self):

        # I don't think we necessarily need to sort each and every one of these lists. Just the min() and the max()
        # will do. We can remove these once we are trying to optimize. For now, I'm doing exactly how the kotlin one
        # does.

        numLinks = sorted([len(node.links) for node in self.nodes])
        numNeighbors = sorted([len([link.otherThan(node) for link in node.links]) for node in self.nodes])
        linkLengths = sorted([(link.node1.loc - link.node2.loc) for link in self.links])
        linkSuccPossibilities = sorted([pow(math.e, -self.alpha + linkLength) for linkLength in linkLengths])
        numQubits = sorted([node.nQubits for node in self.nodes])

        avgLinks = sum(numLinks) / self.n
        avgNeighbors = sum(numNeighbors) / self.n
        avgLinkLength = sum(linkLengths) / len(self.links)
        avglinkSuccP = sum(linkSuccPossibilities) / len(self.links)
        avgQubits = sum(numQubits) / self.n

        # I didn't understand why the .format() is used here. I've rounded them off to the decimal places provided in
        # the parameter S: ! I'm not sure that this string formatting is correct in Python. The $ is used in Kotlin
        # to introduce variables values
        return f"""
              Topology:
              {self.n} nodes, {len(self.links)} links         alpha: {self.alpha}  q: {self.q}
              #links     per node                (Max, Avg, Min): {numLinks[-1]}                        {round(avgLinks, 2)}        {numLinks[0]}
              #qubits    per node                (Max, Avg, Min): {numQubits[-1]}                           {round(avgQubits, 2)}       {numQubits[0]}
              #neighbors per node                (Max, Avg, Min): {numNeighbors[-1]}                    {round(avgNeighbors, 2)}    {numNeighbors[0]} 
              length of links       (km)         (Max, Avg, Min): {round(linkLengths[-1], 2)}               {round(avgLinkLength, 2)}   {round(linkLengths[0], 2)} 
              P(entanglement succeed for a link) (Max, Avg, Min): {round(linkSuccPossibilities[-1], 2)} {round(avglinkSuccP, 2)}    {round(linkSuccPossibilities[0], 2)}
              """

    # Function to get the k-hop neighbors in the network.
    def kHopNeighbors(self, root, k):
        # In case we have a bigger k than the given in the initial conditions, we return all the nodes.
        if k > self.k: return set(self.nodes)
        # A list of false values with the length of nodes list.
        registered = [False for _ in self.nodes]
        # We create a stack and we push the root. We also set the root as a visited node.
        stack = [root]
        registered[root.id] = True

        def work():
            # We take the current node of the stack.
            current = stack[-1]
            # We go recursively through the stack if its size is equal or smaller than k+1.
            if len(stack) <= k + 1:
                # From the current node, we take all the nodes it has links with. We then take the ones that are not yet
                # registered and build a set out of them.
                unregisteredNeighbors = set(filter(lambda x: x.id not in registered,
                                                   [Edge(link.node1, link.node2).otherThan(current) for link in
                                                    current.links]))

                for unregisteredNeighbor in unregisteredNeighbors:
                    # We check the neighbor as visited.
                    registered[unregisteredNeighbor.id] = True
                    # We push the neighbor to the stack so that we can then evaluate the neighbor's neighbors.
                    stack.append(unregisteredNeighbor)
                    # We call recursively this function.
                    work()
                    # We pop the current node from the stack.
                    stack.pop()

        work()
        # We return a list of the k neighbors.
        res = []
        for idx, val in enumerate(registered):
            if val:
                res.append(self.nodes[idx])
            else:
                res.append(self.sentinal)

        res = set(filter(lambda x: x != self.sentianl, res))

        return res

    # It does the same as the previous function, but instead of returning a list of the k-neighbors, it returns all the
    # links in the range.
    def kHopNeighborLinks(self, root, k):
        registered = [False for _ in self.nodes]
        result = set()

        stack = [root]
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

    # Function to get the established entanglements on the topology.
    def getEstablishedEntanglements(self, node1, node2):
        # We create a stack of pairs: <link, node>
        # We push n1 to the stack.
        stack = [(None, node1)]
        # Result is a list of paths, which are lists of nodes.
        result = []

        while stack:
            incoming, current = stack.pop()
            # We create a list with just one element: n2
            if current == node2:
                path = [node2]
                ### As there may be an error, maybe we can use a try-catch
                try:
                    inc = incoming
                except:
                    raise RuntimeError('No such incoming node')
                # For the incoming link, while its n1 and n2 aren't the n1 we are studying, we take the previous node.
                # The previous node is either n2 (if the link's n1 is the last of the path) or n1 otherwise.
                # We update inc to be the first (next) internal link that is not inc.
                while inc.node1 != node1 and inc.node2 != node2:
                    prev = inc.node2 if inc.node1 == path[-1] else inc.node1
                    inc = [link.otherThan(inc) for link in prev.internalLinks if inc in link]
                    path.append(prev)
                # We add n1 to the path and reverse what we found
                path.append(node1)
                result.append(list(reversed(path)))
                continue
            # The if case happens in the first iteration. We take all the links that are not entangled/swapped.
            # The second case chooses all links that contain the incoming one and maps them to the other side.
            outgoingLinks = list(filter(lambda link: link.entangled and not link.swappedAt(current),
                                        current.links)) if incoming is None else [link.otherThan(incoming) for link in
                                                                                  list(filter(lambda internalLink:
                                                                                              incoming in internalLink,
                                                                                              current.internalLinks))]
            # We push every link to the stack to be evaluated.
            # The node that is pushed is the "other side" of current.
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



## Adding this part to run some basic tests on the generate() method.

if __name__ == "__main__":
    generate(50, .9, 5, .1, 6)

## Status:
## Testing incomplete due to the lack of util function implementation.
## Made minor changes to the code with clear comments. Please watch out for '##' which means I added the comment.
## Testing can be completed within 1 or 2 days subject to the util function implementation.
