import math
import random


class Link:
    count = 0

    def __init__(self,
                 topo,
                 node1,
                 node2,
                 l,
                 entangled=False,
                 swap1=False,
                 swap2=False):
        self.swap2 = swap2
        self.swap1 = swap1
        self.l = l
        self.node2 = node2
        self.node1 = node1
        self.topo = topo
        self.entangled = entangled
        Link.count += 1
        self.id = Link.count
        self.assigned = False

    def theOtherEndOf(self, node):
        if self.node1 == node:
            return self.node2
        elif self.node2 == node:
            return self.node1
        else:
            raise RuntimeError('No such node')

    def contains(self, node): return self.node1 == node or self.node2 == node
    def swappedAt(self, node): return (self.node1 == node and self.swap1) or (self.node2 == node and self.swap2)
    def swappedAtTheOtherEndOf(self, node): return (self.node1 == node and self.swap2) or (self.node2 == node and self.swap1)
    def swapped(self): return self.swap1 or self.swap2
    def notSwapped(self): return not self.swapped(self)


    def hashCode(self): return self.id
    def equals(self, other): return other is not None and type(other) is Link and other.id == self.id

    def assignQubitsUtil(self, value):
        if self.assigned == value:
            return
        if value:
            self.node1.remainingQubits -= 1
            self.node2.remainingQubits -= 1
        else:
            self.node1.remainingQubits += 1
            self.node2.remainingQubits += 1
        self.assigned = value
        assert self.node1.remainingQubits >= 0 and self.node1.remainingQubits <= self.node1.nQubits
        assert self.node1.remainingQubits >= 0 and self.node1.remainingQubits <= self.node1.nQubits

    def assignQubits(self):
        value = True
        self.assignQubitsUtil(value)


    def tryEntanglement(self):
        p, b = math.exp(-self.topo.alpha * self.l), self.assigned
        self.entangled = b and p >= random.uniform(0, 1)
        return self.entangled

    def clearEntanglement(self):
        value = False
        self.assignQubitsUtil(value)
        self.entangled = False



	# Need to figure out the string manipulation part
	
    # def toString(self):	pass
     

    def assignable(self):
        return not self.assigned and self.node1.remainingQubits > 0 and self.node3.remainingQubits > 0

	









if __name__ == '__main-__':
    print(Link.count)



