"""This class represents the base for the implementation of a quantum routing algorithm """
from ..topo import Topo


class Algorithm:
    def __init__(self, topo):
        self.topo = topo
        self.name = ''
        self.settings = 'Simple'
        # TODO: How to declare a log writer in Python?
        self.logWriter = ''
        # This is a list of pairs of nodes: Pair<Node,Node>
        self.srcDstPairs = []

    def work(self, pairs):
        assert self.topo.isClean()
        self.srcDstPairs.clear()
        self.srcDstPairs.extend(pairs)
        for p in pairs: logWriter.write("{}⟷{}".format(p[0], p[1]))
        self.P2()
        self.tryEntanglement()
        self.P4()
        established = []
        for p in self.srcDstPairs:
            n1, n2 = p[0], p[1]
            established.append(((n1, n2), self.topo.getEstablishedEntanglements(n1, n2)))
        string = f"""[{}] Established:""".format(self.settings)
        for el in established:
            nodes, length = el[0], len(el[1])
            n1, n2 = nodes[0], nodes[1]
            string.join("{}⟷{} × {}".format((n1, n2, length)))
        string.join(' - {}'.format(self.name))
        print(string)
        self.topo.clearEntanglements()
        # TODO: Figure out how to do this filter
        countNotEmpty = list(filter())
        sumByLength = sum(len(p[1]) for p in established)
        return countNotEmpty, sumByLength

    # Tries an entanglement in each of the links of the topology
    def tryEntanglement(self):
        for link in self.topo.links:
            link.tryEntanglement()

    # TODO: Figure out abstract function implementation
    def prepare(self):
        return ''

    def P2(self):
        return ''

    def P4(self):
        return ''
