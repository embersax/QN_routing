# to be completed with Ruilin's modified version

from QN_routing.algorithm.AlgorithmBase import Algorithm
from QN_routing.topo.Topo import Topo


class GreedyGeographicRouting(metaclass=Algorithm):
    def __init__(self, topo):
        self.topo = topo
        # overrides 'name' from Algorithm class
        self.name = 'Greedy_G'
        self.pathsSortedDynamically = []


    def prepare(self):
        assert self.topo.isClean()

    def P2(self):
        # clearing pahtsSortedDynamically
        pass

    def P4(self):
        pass


