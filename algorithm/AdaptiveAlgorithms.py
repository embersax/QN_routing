from algorithm.AlgorithmBase import Algorithm
from topo.Topo import *
from utils.utils import *
from utils.Disjoinset import *
import heapq

class OfflinePathBasedAlgorithm(Algorithm):

    def __init__(self,topo,allowRecoveryPaths=True):
        super().__init__(topo)
        self.pathsSortedDynamically=[]
        self.extraPaths=[]
        self.fStateMetric=ReducibleLazyEvaluation(initlizar, pre, post)
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



