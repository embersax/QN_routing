from .AlgorithmBase import Algorithm
from ..Topo import Topo
from ..Node import to
import abc


class OnlineAlgorithm(Algorithm):
    def __init__(self, topo, allowRecoveryPaths=True):
        self.topo = topo
        self.allowRecoveryPaths = allowRecoveryPaths
        self.name = 'Online'.join('-R' if allowRecoveryPaths else '')
        # This is a list of PickedPaths
        self.majorPaths = []
        #  HashMap<PickedPath, LinkedList<PickedPath>>()
        self.recoveryPaths = {}

    def prepare(self):
        pass

    def P2(self):
        assert self.topo.isClean()
        self.majorPaths.clear()
        self.recoveryPaths.clear()
        # IMPORTANT: This is not implemented yet. It uses the lay evaluation function.
        self.pathToRecoveryPaths.clear()
        while True:
            candidates = self.calCandidates(self.srcDstPairs)
            pick = max(candidates[0])
            if pick is not None and pick[0] > 0.0:
                self.pickAndAssignPath(pick)
            else:
                break
        if self.allowRecoveryPaths:
            self.P2Extra()

    def P2Extra(self):
        for majorPath in self.majorPaths:
            _, _, p = majorPath
            # Not sure if we have to make these functions inclusive for the upper range (i.e. +1)
            for l in range(1, self.topo.k):
                for i in range(0, len(p) - l - 1):
                    (src, dst) = to(p[i], p[i + l])
                    candidates = self.calCandidates(list(tuple(src, dest)))
                    pick = max(candidates[0])
                    if pick is not None and pick[0] > 0.0:
                        self.pickAndAssignPath(pick, majorPath)

    def P4(self):
        pass

    # TODO: we need to implement these functions
    def calCandidates(self, srcDstPairs):
        return ''

    def pickAndAssignPath(self, pick):
        return ''
