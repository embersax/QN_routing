from .AlgorithmBase import Algorithm
from QN_routing.topo.Topo import Topo, Path
from QN_routing.topo.Node import to
from QN_routing.utils.CollectionUtils import ReducibleLazyEvaluation
import abc
from dataclasses import dataclass
from functools import reduce
from itertools import dropwhile

@dataclass
class RecoveryPath:
    path: Path
    width: int
    taken: int = 0
    available: int = 0

# is super() call not necessary here?
class OnlineAlgorithm(Algorithm):
    def __init__(self, topo, allowRecoveryPaths=True):
        self.topo = topo
        self.allowRecoveryPaths = allowRecoveryPaths
        self.name = 'Online'.join('-R' if allowRecoveryPaths else '')
        # This is a list of PickedPaths
        self.majorPaths = []
        #  HashMap<PickedPath, LinkedList<PickedPath>>()
        self.recoveryPaths = {}
        self.pathToRecoveryPaths = []
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
                    candidates = self.calCandidates(list(tuple(src, dst)))
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


    def P4(self):

        for pathWithWidth in self.majorPaths:
            _, width, majorPath = pathWithWidth
            oldNumPairs = len(self.topo.getEstablishedEntanglements(majorPath[0], majorPath[-1]))
            recoveryPaths = sorted(self.recoveryPaths[pathWithWidth], key=lambda tup: len(tup[2])*10000 + majorPath.index(tup[2][0]))

            for _, w, p in recoveryPaths:
                available = min([len([link.contains(node2) and link.entangled for link in node1.links]) for node1, node2 in Path.edges(p)])
                self.pathToRecoveryPaths[pathWithWidth].append(RecoveryPath(p, w, 0, available))

            edges = list(zip(range(len(majorPath)-1), range(1, len(majorPath))))
            rpToWidth = {recPath[2]: recPath[1] for recPath in recoveryPaths}

            for i in range(1, width+1):

                def filterForBrokenEdges(tup):
                    i1, i2 = tup
                    n1, n2 = majorPath[i1], majorPath[i2]
                    checkAny = [link.contains(n2) and link.assigned and link.notSwapped() for link in n1.links]
                    return any(checkAny)

                brokenEdges = list(filter(filterForBrokenEdges, edges))
                edgeToRps = {brokenEdge: [] for brokenEdge in brokenEdges}
                rpToEdges = {recPath[2]: [] for recPath in recoveryPaths}

                for _, _, rp in recoveryPaths:
                    s1, s2 = majorPath.index(rp[0]), majorPath.index(rp[-1])
                    reqdEdges = list(filter(lambda edge: edge in brokenEdges, list(zip(range(s1, s2), range(s1+1, s2+1)))))

                    for edge in reqdEdges:
                        rpToEdges[rp] = edge
                        edgeToRps[edge] = rp

                realPickedRps = {}
                realRepairedEdges = {}

                # Try to cover the broken edges

                for brokenEdge in brokenEdges:
                    if brokenEdge in realRepairedEdges:
                        continue
                    repaired = False
                    next = 0

                    toBeDone = True # a dummy variable to indicate the gaps and for indentation
                    while toBeDone:
                        # Not sure what this block is supposed to do.
                        # tryRp block begin#
                        repairedEdges = set()
                        pickedRps = set()
                        # tryRp block end#


                        repaired = True
                        repairedEdges.add(brokenEdge)
                        pickedRps.add(rp)

                        for item1, item2 in zip(realPickedRps, pickedRps):
                            item = item1 - item2
                            rpToWidth[item] += 1
                            item = -item
                            rpToWidth[item] -= 1

                        realPickedRps = pickedRps
                        realRepairedEdges = repairedEdges
                        break

                    if not repaired:
                        break

                def doInFold(acc, rp):
                    idx = -1
                    for ele in self.pathToRecoveryPaths[pathWithWidth]:
                        if ele.path == rp:
                            idx = self.pathToRecoveryPaths[pathWithWidth].index(ele)

                    pathData = self.pathToRecoveryPaths[pathWithWidth][idx]
                    pathData.taken += 1
                    toAdd = Path.edges(rp)
                    toDelete = Path.edges(list(
                        dropwhile(lambda it: it != rp[-1], list(reversed(list(dropwhile(lambda it: it != rp[0], acc)))))))
                    edgesOfNewPathAndCycles = set(Path.edges(acc)).difference(toDelete).union(toAdd)

                    # our implementation of ReducibleLazyEvaluation requires 2 inputs K and V but Shoqian initialized it with just one. Has to be looked at.
                    p = self.topo(edgesOfNewPathAndCycles, acc[0], acc[-1], ReducibleLazyEvaluation(1.0))
                    return p

                def foldLeft(realPickedRps, majorPath):
                    return reduce(doInFold, realPickedRps, majorPath)

                p = foldLeft(realPickedRps, majorPath)

                zippedP = list(zip(list(zip(p[:-2], p[1:]))[:-1], p[2:]))

                for n12, next in zippedP:
                    prev, n = n12

                    prevLinks = list(sorted(filter(lambda it: it.entangled and it.swappedAt(n) and prev in it, n.links), key=lambda it:it.id))[0]
                    nextLinks = list(sorted(filter(lambda it: it.entangled and it.swappedAt(n) and next in it, n.links), key=lambda it:it.id))[0]

                    prevAndNext = list(zip(prevLinks, nextLinks))
                    for l1, l2 in prevAndNext:
                        n.attemptSwapping(l1, l2)

            succ = len(self.topo.getEstablishedEntanglements(majorPath[0], majorPath[-1])) - oldNumPairs
            self.logWriter.write("""{}, {} {}""".format([it.id for it in majorPath], width, succ))
            for it in self.pathToRecoveryPaths[pathWithWidth]:
                self.logWriter.write("""{}, {} {} {}""".format([it2.id for it2 in it.path], width, it.available, it.taken))


        self.logWriter.write("\n")

