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
        self.fStateMetric=ReducibleLazyEvaluation( initilizer=lambda x : length(list_minus(x.n1.loc,x.n2.loc))+topo.internalLength)
        def initlizar( src,dst) :
            q = []
            # havn't finished edges
            tmp1 = groupby_dict( lambda x: to(x.n1,x.n1)   ,topo.links)
            tmp2 = list( map(  lambda x: to(tmp1[x].n1 ,tmp1[x].n2 ),tmp1) )
            edges = set (tmp2)
            visited = {}
            p = topo.shortestPath(edges, src ,dst ,self.fStateMetric)
            visited [edges] = p
            if p[1] is not None:
                if p not in q:
                    heapq.heappushpop(q,(p[0],p) )

            def work():
                tmp_list = list(visited)
                for edge, p  in tmp_list:
                    if p[1] is None:
                        return
                    relatedEdges = set(p[1].edges())
                    for edge in relatedEdges:
                        edges.remove(edge)
                        if edge not in visited:
                            p=topo.shortestPath(edges,src,dst,self.fStateMetric)
                            visited[edges] = p
                            if p[1] is not None and p not in q:
                                heapq.heappushpop(q, (p[0], p))
                        edges.add(edge)
            tries = 0
            while ( len(q)<50 and (tries < 100)):
                work()
                tries+=1
            tmp = []
            while (q is not []):
                p = q.pop(0)
                width = topo.widthPhase2(p[1])
                tmp.append( [p[0],width,p[1]])
            return tmp
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
    def prepare(self):
        assert self.topo.isClean()



