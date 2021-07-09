from .AlgorithmBase import Algorithm
from ..Topo import Topo
from ..Node import to
import numpy as np
import heapq
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
                a, b, c = set(topo.nodes), set(src), set(dst)
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
                heapq.heappush(queue, (priority(to(src, self.topo.sentinal)), to(src, self.topo.sentinal)))
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
                            heapq.heappush(queue, (priority(to(neighbor, u)), to(neighbor, u)))

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
