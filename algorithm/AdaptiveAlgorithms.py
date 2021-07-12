from QN_routing.algorithm.AlgorithmBase import Algorithm
from QN_routing.topo.Topo import *
from QN_routing.utils.utils import *
from QN_routing.utils.Disjoinset import *
from QN_routing.utils.CollectionUtils import PriorityQueue
import math
from abc import ABC


# returns the priority based on the type and the element
# trying to use this as a replacement for the dynamic comparator
def getPriority(element, type=None):
    if type is None or type in {'SumDist', 'CR'}:
        return element
    elif type == 'MultiMetric':
        return math.log(element[0], 2) - element[1] + 10000 if element[1] == 1 else math.log(element[0], 2) - element[
            1] + 0
    elif type == 'BotCap':
        return element[1] * 10000 + element[0]

    else:
        raise RuntimeError('Invalid priority type')


class OfflinePathBasedAlgorithm(Algorithm, ABC):

    def __init__(self, topo, allowRecoveryPaths=True):
        super().__init__(topo)
        self.pathsSortedDynamically = []
        self.extraPaths = []
        self.priorityType = None

        # self.fStateMetric=ReducibleLazyEvaluation(initlizar, pre, post)
        def initlizar(src, dst):
            q = []
            # havn't finished edges
            edges = groupby_dict()
            visited = {}
            p = topo.shortestPath(edges, src, dst, self.fStateMetric)

        def pre(src, dst):
            if (src.id > dst.id):
                return (dst, src)
            else:
                return (src, dst)

        def post(src, dst, path):
            if src.id > dst.id:
                return path.reverse()
            else:
                return path

        self.allowRecoveryPaths = True
        self.pathsSortedStatically = ReducibleLazyEvaluation(initlizar, pre, post)



    # to be completed
    def P2(self):
        self.pathsSortedStatically.clear()
        self.extraPaths.clear()

        allPaths = [self.pathsSortedStatically[it] for it in self.srcDstPairs]

        # this queue needs to have tuples in it with the tup[0] being the actual priority
        # https://www.pythonpool.com/python-priority-queue/
        # I think it is okay to trade some space off for getting comparators out of the way.
        # The logic can be implemented when we push the elements to the queue with a function call to get the priority of the element

        q, type = PriorityQueue(), self.priorityType
        for path in allPaths:
            q.push(path, getPriority(path))

        cnt = 0

        while q and cnt < 100 * len(self.srcDstPairs):
            p = q.pop()
            width = self.topo.widthPhase2(p[3])

            if width == 0:
                q.push(p, getPriority(p))
                break

            if width < p[1]:
                newP = (p[0], width, p[2])
                q.push(newP, getPriority(newP))
                continue

            self.pathsSortedDynamically.append(p)

            for i in range(1, width + 1):
                for node1, node2 in Path.edges(p[2]):
                    for link in node1.links:
                        if (link.node1 == node2 or link.node2 == node2) and link.assigned:
                            link.assignQubits()
                            break

        self.P2Online(q)

        if self.allowRecoveryPaths:
            self.P2Recovery(q)

    def P2Recovery(self, q):
        pass

    def P2Online(self, q):
        pass


class SumDist(OfflinePathBasedAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        super().__init__(topo, allowRecoveryPaths)
        self.name = 'SumDist-R' if allowRecoveryPaths else 'SumDist'
        self.priorityType = 'SumDist'


class CreationRate(OfflinePathBasedAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        super().__init__(topo, allowRecoveryPaths)
        self.name = 'CR-R' if allowRecoveryPaths else 'CR'
        # not sure what n1, n2 and it are in shoqian's code for fStateMetric in this class
        locDifference = 1.0
        self.fStateMetric = ReducibleLazyEvaluation(math.exp(self.topo.alpha * length(locDifference)))
        self.priorityType = 'CR'


class MultiMetric(OfflinePathBasedAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        super().__init__(topo, allowRecoveryPaths)
        self.name = 'MultiMetric-R' if allowRecoveryPaths else 'MultiMetric'
        self.priorityType = 'MultiMetric'


class BotCap(OfflinePathBasedAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        super().__init__(topo, allowRecoveryPaths)
        self.name = 'BotCap-R' if allowRecoveryPaths else 'BotCap'
        self.priorityType = 'BotCap'


# this class is not used anywhere
class GlobalLinkState():
    pass