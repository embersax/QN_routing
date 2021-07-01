import math
import random
import sys
import re
class Topo:
    def __init__(self,input):
        self.n=0
        self.alpha=0
        self.q=0
        self.k=0
        self.nodes=[]
        self.links=[]
        sefl.sential=None
        self.nodeDigits=0
        self.linkDigits=0
        self.distanceDigits=0
        self.internalLength=0

        lines= list(filter(lambda x: x!="" ,re.sub("""\s*//.*$""","", input.splitlines())))
        n=int(lines.pop(0))
        self.sential=Node(self,-1,[-1,-1,-1],-sys.maxint)
        self.alpha=float(lines.pop(0))
        self.q=float(line.pop(0))
        self.k=int(lines.pop(0))
        self.internalLength=math.log(1/q)/alpha

        for i in range(n):
            line=lines.pop(0)
            tmp=list(map(lambda x: int(x) ,line.split(" ")))
            node=Node(self,i,[tmp[1],tmp[2]],int(tmp[0]))
            self.nodes.append(node)

        self.nodeDigits=round(math.ceil(math.log10(float(len(nodes)))))

        while len(lines):
            linkLine=lines.pop(0)
            tmp=list(map(lambda x: int(x) ,line.split(" ")))
            tmp1=tmp[0:2]
            tmp2=list(map(lambda x: nodes[x],tmp1))
            n1,n2=sorted(tmp2, key=lambda x: x.id, reverse=True)
            nm=tmp1[2]

            if n1.id >=n2.id:
                sys.exit()
            for i in range(1,nm):
                link=Link(self,n1,n2,+(n1.loc - n2.loc))
                self.links.append(link)
                n1.links.append(link)
                n2.links.append(link)
        self.linkDigits=round(math.ceil(math.log10(len(links))))
        flat_map = lambda f, xs: (y for ys in xs for y in f(ys))
        self.distanceDigits=round(math.ceil(math.log10( max(list(map(lambda x: x.l))+list(map(lambda x : math.abs(x),list(flat_map(lambda x: x.loc, nodes))))))))













