from algorithm.AlgorithmBase import Algorithm
from topo.Topo import *
from utils.utils import *
from utils.Disjoinset import *

class OfflinePathBasedAlgorithm(Algorithm):

    def __init__(self,topo,allowRecoveryPaths=True):
        self.allowRecoveryPaths=True
        self.