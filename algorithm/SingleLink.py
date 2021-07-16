# It looks like single link is an instantiation of the OnlineAlgorithm class.
from .OnlineAlgorithm import OnlineAlgorithm
from ..topo.Topo import Topo
from ..topo.Node import to

class SingleLink(OnlineAlgorithm):
    def __init__(self, topo, allowRecoveryPaths):
        self.topo.links = topo.links

