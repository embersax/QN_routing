import math
import random
import sys
import re
from itertools import combinations, groupby

from QN_routing.topo.Link import Link
from QN_routing.topo.Node import Node


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

        ## abs is defined in the standard library itself. Replacing math.abs with abs
        self.distanceDigits = round(math.ceil(math.log10(
            max(list(map(lambda x: x.l, self.links)) +
                list(map(lambda x: abs(x), list(flat_map(lambda x: x.loc, self.nodes))))))))

    @staticmethod
    def listtoString(s, seperator):
        s = list(map(str(), s))
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
                oldP[m] = P[m]

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
        element = [random.uniform(0, 1) * 100 for _ in range(2)]
        if all(sum(x - element) > controllingD / 1.2 for x in nodeLocs):
            nodeLocs.append(element)
    nodeLocs = sorted(nodeLocs, key=lambda x: x[0] + int(x[1] * 10 / edgeLen) * 1000000)

    def argument_function(beta):

        # combination function needed to be finished
        ## An efficient algorithm is already available in itertools. Replacing with combinations() call
        tmp_list = combinations([i for i in range(0, n)])
        for i in range(len(tmp_list)):
            (l1, l2) = list(map(lambda x: nodeLocs[x], [tmp_list[i][0], tmp_list[i][1]]))
            # currentlty not sure about how "+" computes
            d = sum(l2 - l1)
            if d < 2 * controllingD:
                l = min([random.uniform(0, 1) for i in range(50)])
                r = math.exp(-beta * d)
                if l < r:
                    # to functions needed to be implemented
                    links.append(tmp_list[i][0].to(tmp_list[i][1]))
        return 2 * float(len(links)) / n

    # Can't fully debug unless the dynSearch and disjointSet are actually implemented
    # dynSearch needed to be implememnted
    beta = dynSearch(0, 20, float(degree), False, 0.2)
    # DisjoinSet needed to be implemmented
    disjoinSet = DisjointSet(n)
    for i in range(len(links)):
        disjoinSet.merge(links[i][0], links[i][1])
    # compute  ccs: =(o unitil n).map.map.groupby.map.sorted
    t2 = list( map( lambda id,p:[id,p] ,
                    list(map( lambda x:x.to(disjoinSet.getgetRepresentative(x))
                    , [i for i in range(0,n)] ))))
    t3=self.grouby_dict(t2,lambda x: x[1])
    t4=list(map(lambda x: list(map(lambda x:x[0] ,t3[x]) ) ,t3))
    ccs =sorted(t4,lambda x:-len(x))

    biggest = ccs[0]
    for i in range(1, len(ccs)):
        # this part I'm not sure since shuffle changes the order of original ccs list
        ## ccs1 is not defined. Please check
        tmp1 = random.shuffle(ccs1)[0:3]
        for j in range(len(tmp1)):
            nearest = sorted(biggest, key=lambda x: sum(nodeLocs[x] - nodeLocs[tmp1[j]]))[0]
            tmp2 = sorted([nearest, tmp1[j]])
            links.append(tmp2[0].to(tmp2[1]))

    ## groupby_dict, listtoString are functions Topo class objects. They seem to be used outside the class. So, making them static in this case.
    flat_map = lambda f, xs: (y for ys in xs for y in f(ys))
    #     retrive the flatten list first
    tmp_list = list(flat_map(lambda x: [x[0], x[1]], links))
    # retriev dictionary to iterate
    tmp_dict = Topo.grouby_dict(tmp_list, lambda x: x)
    for key in tmp_dict:
        if len(tmp_dict[key]) / 2 < 5:
            # not sure if takes the right index, needed to double check
            nearest = sorted([i for i in range(0, n)], lambda x: sum(nodeLocs[x] - nodeLocs[key]))[
                      1:6 - len(tmp_dict[key]) / 2]
            tmp_list = list(map(lambda x: [sorted[x, key][0], sorted[x, key][1]], nearest))
            # need to check what is added to links here
            for item in tmp_list:
                links.append(item)
    nl = "\n"
    tmp_string1 = Topo.listtoString(
        list(map(lambda x: f"{int(random.uniform(0, 1) * 5 + 10)}" + Topo.listtoString(x, " "), nodeLocs)), nl)
    tmp_string2 = Topo.listtoString(
        list(map(lambda x: f"{x[0]}" + f"{x[1]}" + f"{int(random.uniform(0, 1) * 5 + 3)}", links)), nl)
    return Topo(f"""
            {n}
            {alpha}
            {q}
            {k}
            {tmp_string1}
            {tmp_string2}
"""

                )


## Adding this part to run some basic tests on the generate() method.

if __name__ == "__main__":
    generate(50, .9, 5, .1, 6)

## Status:
## Testing incomplete due to the lack of util function implementation.
## Made minor changes to the code with clear comments. Please watch out for '##' which means I added the comment.
## Testing can be completed within 1 or 2 days subject to the util function implementation.
