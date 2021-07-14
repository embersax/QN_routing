from .AlgorithmBase import Algorithm
from topo.Topo import Topo, Path
from topo.Node import to
import numpy as np
import heapq
from utils.utils import ReducibleLazyEvaluation
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
        super().__init__(topo)
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
            # maxby first element
            pick = max(candidates,key=lambda x: x[0])
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
                    pick = max(candidates,key=lambda x: x[0])
                    if pick is not None and pick[0] > 0.0:
                        self.pickAndAssignPath(pick, majorPath)



    # ops is a list of pairs of nodes
    # TODO: There's something weird call pmap in the kotlin code. I'm not sure but it looks it's not important, as it
    #  basically iterates over the pairs of nodes of the list.
    # This function returns a list of picked path: a triple
    def calCandidates(self, ops):
        for o in ops:
            src, dst = o[0], o[1]
            maxM = min(src.remainingQubits, dst.remainingQubits)
            if maxM == 0: return None
            # Candidate should be of type PickedPath, which is a (Double,Int,Path) triple.
            candidate = None
            # In the kotlin code it goes until 1 but, to include 1, we must set the range parameter to 0
            for w in range(maxM, 0, -1):
                a, b, c = set(self.topo.nodes), set(src), set(dst)
                # We subtract the intersection from the union to get the difference between the three sets.
                tmp = (a | b | c) - (a & b & c)
                # In the kotlin code, it's a hashset, but I think a set works fine.
                failNodes = set([node for node in tmp if node.remainingQubits < 2 * w])
                tmp0 = [link for link in self.topo.links if
                        not link.assigned and link.node1 not in failNodes and link.node2 not in failNodes]
                tmp1 = {}
                for link in tmp0:
                    if (link.node1, link.node2) in tmp1:
                        tmp1[link].append(link)
                    else:
                        tmp1[(link.node1, link.node2)] = [link]
                # So I think we do not need the filter part if we do it this way. We only need the edges.
                edges = set([edge for edge in tmp1 if len(tmp1[edge]) >= w])
                # TODO: ReducibleLazyEvaluation part
                # key: node, value: list of nodes
                neighborsOf = {}
                for edge in edges:
                    if edge.node1 in neighborsOf:
                        neighborsOf[edge.node1].append(edge.node2)
                    else:
                        neighborsOf[edge.node1] = [edge.node2]
                    if edge.node2 in neighborsOf:
                        neighborsOf[edge.node2].append(edge.node1)
                    else:
                        neighborsOf[edge.node2] = [edge.node1]
                if len(neighborsOf[src]) == 0 or len(neighborsOf[dst]) == 0: continue
                # This is a hashmap of nodes: <Node,Node>
                prevFromSrc = {}

                def getPathFromSrc(n):
                    # This is a list of nodes
                    path = []
                    cur = n
                    while cur != self.topo.sentinal:
                        path.insert(0, cur)
                        cur = prevFromSrc[cur]
                    return path

                def priority(edge):
                    node1, node2 = edge.node1, edge.node2
                    if E[node1.id][0] < E[node2.id][0]:
                        return 1
                    elif E[node1.id][0] == E[node2.id][0]:
                        return 0
                    else:
                        return -1

                E = {-float('inf'): np.zeros(w + 1) for _ in range(len(self.topo.nodes))}
                # TODO: Ask about the priority queue implementation
                q = []
                E[src.id] = (float('inf'), np.zeros(w + 1))
                heapq.heappush(q, (priority(to(src, self.topo.sentinal)), to(src, self.topo.sentinal)))
                while q:
                    u, prev = q[0], q[1]  # invariant: top of q reveals the node with highest e under m
                    if u in prevFromSrc: continue  # skip same node suboptimal paths
                    prevFromSrc[u] = prev  # record

                    if u == dst:
                        candidate = (E[u.id][0], w, getPathFromSrc(dst))
                        break

                    for neighbor in neighborsOf[u]:
                        temp = E[u.id][1].clone()
                        e = self.topo.e(getPathFromSrc(u) + neighbor, w, temp)
                        newE = (e, temp)
                        oldE = E[neighbor.id]

                        if oldE[0] < newE[0]:
                            E[neighbor.id] = newE
                            heapq.heappush(q, (priority(to(neighbor, u)), to(neighbor, u)))

                if candidate is not None: break
        return [c for c in candidate if c is not None]


    # The pick variable is a (Double, Int, Path) triple. Recall that a path is a list of nodes.
    def pickAndAssignPath(self, pick, majorPath=None):
        if majorPath is not None:
            self.recoveryPaths[majorPath] = pick
        else:
            self.majorPaths.append(pick)
            self.recoveryPaths[pick] = []
        # We get the second value of the triple
        width = pick[1]
        # The first argument of toAdd is a link bundle: list of links. The third argument is a map, where the key is of
        # type Edge and the values are lists of pairs. The pairs are made of connections (lists of link bundles).
        toAdd = ([], width, {})

        for edge in pick[2].edges():
            n1, n2 = edge[0], edge[1]
            links = sorted([link for link in n1.links if not link.assigned and link.contains(n2)],
                           key=lambda link: link.id)[:width]
            assert len(links) == width
            toAdd[0].append(links)
            for link in links:
                link.assignQubits()
                link.tryEntanglement()  # just for display

    def P4(self):

        for pathWithWidth in self.majorPaths:
            _, width, majorPath = pathWithWidth
            oldNumPairs = len(self.topo.getEstablishedEntanglements(majorPath[0], majorPath[-1]))
            recoveryPaths = sorted(self.recoveryPaths[pathWithWidth],
                                   key=lambda tup: len(tup[2]) * 10000 + majorPath.index(tup[2][0]))

            for _, w, p in recoveryPaths:
                available = min(
                    [len([link.contains(node2) and link.entangled for link in node1.links]) for node1, node2 in
                     Path.edges(p)])
                self.pathToRecoveryPaths[pathWithWidth].append(RecoveryPath(p, w, 0, available))

            edges = list(zip(range(len(majorPath) - 1), range(1, len(majorPath))))
            rpToWidth = {recPath[2]: recPath[1] for recPath in recoveryPaths}

            for i in range(1, width + 1):

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
                    reqdEdges = list(
                        filter(lambda edge: edge in brokenEdges, list(zip(range(s1, s2), range(s1 + 1, s2 + 1)))))

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

                    tryRpContinue = False
                    for rp in list(sorted(list(
                            filter(lambda it: rpToWidth[it] > 0 and not it in realPickedRps, edgeToRps[brokenEdge])),
                                          key=lambda it: majorPath.index[it[0]] * 10000 + majorPath.index[it[-1]])):
                        # there is only a single for loop in this case. So, I don't think the labeled continue makes a difference.
                        if majorPath.index[rp[0]] < next:   continue
                        next = majorPath.index[rp[-1]]
                        repairedEdges = set(realRepairedEdges)
                        pickedRps = set(realPickedRps)

                        otherCoveredEdges = set(rpToEdges[rp]).difference(brokenEdge)

                        for edge in otherCoveredEdges:
                            prevRpSet = set(edgeToRps[edge]).intersection(set(pickedRps)).remove(rp)
                            prevRp = prevRpSet[0] if prevRpSet else None

                            if prevRp == None:
                                repairedEdges.add(edge)

                            else:
                                continue

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
                        dropwhile(lambda it: it != rp[-1],
                                  list(reversed(list(dropwhile(lambda it: it != rp[0], acc)))))))
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

                    prevLinks = list(sorted(filter(lambda it: it.entangled and it.swappedAt(n) and prev in it, n.links),
                                            key=lambda it: it.id))[0]
                    nextLinks = list(sorted(filter(lambda it: it.entangled and it.swappedAt(n) and next in it, n.links),
                                            key=lambda it: it.id))[0]

                    prevAndNext = list(zip(prevLinks, nextLinks))
                    for l1, l2 in prevAndNext:
                        n.attemptSwapping(l1, l2)

            succ = len(self.topo.getEstablishedEntanglements(majorPath[0], majorPath[-1])) - oldNumPairs
            self.logWriter.write("""{}, {} {}""".format([it.id for it in majorPath], width, succ))
            for it in self.pathToRecoveryPaths[pathWithWidth]:
                self.logWriter.write(
                    """{}, {} {} {}""".format([it2.id for it2 in it.path], width, it.available, it.taken))

        self.logWriter.write("\n")
