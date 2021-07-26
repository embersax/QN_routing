from  QN_routing.topo.Topo import Topo
from math import exp
from QN_routing.utils.utils import length
from QN_routing.configs import randGenSeed
from random import shuffle
import random
from itertools import combinations


random.seed(randGenSeed)


def simpleTest():
    netTopology = Topo.generate(50, 0.9, 5, 0.1, 6)

    alphas = []
    p = [.8, .5]
    for expectedAvgP in p:
        alpha = step = .1
        lastAdd = True

        while True:
            lines = list(netTopology.split('\n'))
            lines[1] = str(alpha)
            topo = Topo('\n'.join(lines))
            avgP = float(sum(list(map(lambda it: exp(-alpha + length(it.node1.loc - it.node2.loc)), topo.links)))) / len(topo.links)

            if abs(avgP - expectedAvgP) / expectedAvgP < .001:  break

            elif avgP > expectedAvgP:
                if not lastAdd: step /= 2
                alpha += step
                lastAdd = True

            else:
                if lastAdd: step /= 2
                alpha -= step
                lastAdd = False

        alphas.append(alpha)

    repeat = 1000

    def funInLine135(combs, nsd, repeat):
        res = []
        for i in range(1, repeat+1):
            shuffle(combs)
            res.append(list(map(lambda it: (it[0].id, it[1].id), combs[nsd])))
        return res

    testSetIter, testSet = [i for i in range(1, 11)], []
    for nsd in testSetIter:
        combs = list(combinations(netTopology.nodes, 2))
        testSet.append(funInLine135(combs, nsd, repeat))

    children = []
