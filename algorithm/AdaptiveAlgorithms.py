from QN_routing.algorithm.AlgorithmBase import Algorithm
from QN_routing.topo.Topo import *
from QN_routing.utils.utils import *
from QN_routing.utils.Disjoinset import *
import heapq

class OfflinePathBasedAlgorithm(Algorithm):

    def __init__(self,topo,allowRecoveryPaths=True):
        super().__init__(topo)
        self.pathsSortedDynamically=[]
        self.extraPaths=[]
        # self.fStateMetric=ReducibleLazyEvaluation(initlizar, pre, post)
        def initlizar( src,dst) :
            q = []
            # havn't finished edges
            edges = groupby_dict()
            visited = {}
            p = topo.shortestPath(edges, src ,dst ,self.fStateMetric)

        def pre (src,dst) :
            if (src.id > dst.id):
                return (dst , src)
            else:
                return (src , dst)
        def post (src ,dst , path):
            if  src.id > dst.id :
                return path.reverse()
            else:
                return path
        self.allowRecoveryPaths=True
        self.pathsSortedStatically=ReducibleLazyEvaluation(initlizar, pre, post)

    # to be completed
    def P2(self):
        self.pathsSortedStatically.clear()
        self.extraPaths.clear()

        allPaths = self.srcDstPairs