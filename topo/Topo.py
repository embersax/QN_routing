import math
from .Node import Node
from .Link import Link
import random
import sys
import re


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
            for i in range(1, nm+1):
                link = Link(self, n1, n2, +(n1.loc - n2.loc))
                self.links.append(link)
                n1.links.append(link)
                n2.links.append(link)

        self.linkDigits = round(math.ceil(math.log10(len(self.links))))
        flat_map = lambda f, xs: (y for ys in xs for y in f(ys))
        self.distanceDigits = round(math.ceil(math.log10(
            max(list(map(lambda x: x.l, self.links)) +
                list(map(lambda x: math.abs(x), list(flat_map(lambda x: x.loc, self.nodes))))))))
