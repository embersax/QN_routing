import math
import random


class Link:
    count = 0

    # The constructor of the link. Parameters:
    # A given topology topo.
    # Two nodes (n1 and n2), which form the link.
    # The length l of the link.
    # assigned, which is true if there is one qubit at each end of the link.
    # entangled, which is true if the link corresponds to an entanglement, false otherwise.
    # s1 and s2, which are true if the corresponding extrema of the link have been swapped.
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

    # Given a node n, returns the node at the other end of the link.
    def theOtherEndOf(self, node):
        if self.node1 == node:
            return self.node2
        elif self.node2 == node:
            return self.node1
        else:
            raise RuntimeError('No such node')

    # Checks if the given node n corresponds to either n1 or n2
    def contains(self, node):
        return self.node1 == node or self.node2 == node

    # Checks if there was a swapping at the end of the link that contains node n.
    def swappedAt(self, node):
        return (self.node1 == node and self.swap1) or (self.node2 == node and self.swap2)

    # Checks if there was a swapping on the other end of the link that contains node n.
    def swappedAtTheOtherEndOf(self, node):
        return (self.node1 == node and self.swap2) or (self.node2 == node and self.swap1)

    # Checks if either ends of the link have been swapped.
    def swapped(self):
        return self.swap1 or self.swap2

    # Logical inverse of swapped() function
    def notSwapped(self):
        return not self.swapped(self)

    # Returns the hashcode (id) of the link
    def hashCode(self):
        return self.id

    # Checks if two links are equal
    def equals(self, other):
        return other is not None and type(other) is Link and other.id == self.id

    # Assigns or removes qubits to the extrema of the link depending on the parameter "value".
    # If value == True, we assign qubits to the link. If value == False, we do otherwise.
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
        assert 0 <= self.node1.remainingQubits <= self.node1.nQubits
        assert 0 <= self.node1.remainingQubits <= self.node1.nQubits

    # Assigns qubits to the link
    def assignQubits(self):
        value = True
        self.assignQubitsUtil(value)

    # Tries the execution of an entanglement in the link. First, it calculates the probability p of a successful
    # entanglement, which depends on the length of the channel and the physical parameter alpha. An entanglement is
    # successful if the link has qubits assigned to each end and if the probability p is greater than some uniformly
    # distributed number.
    def tryEntanglement(self):
        p, b = math.exp(-self.topo.alpha * self.l), self.assigned
        self.entangled = b and p >= random.uniform(0, 1)
        return self.entangled

    # "Deletes" an entanglement in this link.
    def clearEntanglement(self):
        value = False
        self.assignQubitsUtil(value)
        self.entangled = False

    # Need to figure out the string manipulation part

    # def toString(self):	pass

    # Checks if we can assign qubits to the link
    def assignable(self):
        return not self.assigned and self.node1.remainingQubits > 0 and self.node3.remainingQubits > 0


if __name__ == '__main-__':
    print(Link.count)
