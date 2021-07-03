import math
import random
import sys
import re
import itertools
class Topo:

    def __init__(self, input):
        self.n = 0
        self.alpha = 0
        self.q = 0
        self.k = 0
        self.nodes = []
        self.links = []
        self.sentinal = None
        self.nodeDigits = 0
        self.linkDigits = 0
        self.distanceDigits = 0
        self.internalLength = 0

        lines = list(filter(lambda x: x != "", re.sub("""\s*//.*$""", "", input.splitlines())))
        n = int(lines.pop(0))
        # Why do we have -sys.maxint and not sys.maxint ?
        self.sentinal = Node(self, -1, [-1, -1, -1], sys.maxint)
        self.alpha = float(lines.pop(0))
        self.q = float(lines.pop(0))
        self.k = int(lines.pop(0))
        self.internalLength = math.log(1 / self.q) / self.alpha

        for i in range(n):
            line = lines.pop(0)
            # I think we need it to be float(x)
            tmp = list(map(lambda x: float(x), line.split(" ")))
            node = Node(self, i, [tmp[1], tmp[2]], int(tmp[0]))
            self.nodes.append(node)

        self.nodeDigits = round(math.ceil(math.log10(float(len(self.nodes)))))

        while len(lines):
            linkLine = lines.pop(0)
            # I think it should be linkLine.split(" ") instead of line.split(" ")
            tmp = list(map(lambda x: int(x), linkLine.split(" ")))
            tmp1 = tmp[0:2]
            tmp2 = list(map(lambda x: self.nodes[x], tmp1))
            # I made a tuple with n1 and n2, as Shouqian did.
            (n1, n2) = sorted(tmp2, key=lambda x: x.id, reverse=True)
            # I think it has to be tmp, as tmp1[2] is None.
            nm = tmp[2]
            assert n1.id < n2.id
            # I'm not sure but maybe as range is not inclusive for the upper limit, it has to be range(1,nm+1),
            # according to Shouqian's code.
            for i in range(1, nm + 1):
                link = Link(self, n1, n2, +(n1.loc - n2.loc))
                self.links.append(link)
                n1.links.append(link)
                n2.links.append(link)

        self.linkDigits = round(math.ceil(math.log10(len(self.links))))
        flat_map = lambda f, xs: (y for ys in xs for y in f(ys))
        self.distanceDigits = round(math.ceil(math.log10(
            max(list(map(lambda x: x.l, self.links)) +
                list(map(lambda x: math.abs(x), list(flat_map(lambda x: x.loc, self.nodes))))))))

    def listtoString(self,s,seperator):
        s=list(map(str(),s))
        return seperator.join(s)
    def toString(self):
        return self.toConfig()
    def grouby_dict(self,s,f):
        tmp={}
        for a, b in itertools.groupby(s, f):
            for thing in b:
                tmp[a]=b
        return tmp


    def toConfig(self):
        nl="\n"
        tmp_dict=self.grouby_dict(self.links,lambda x:x.n1.to(x.n2))
        return f"""
            {self.n}
            {self.alpha}
            {self.q}
            {self.k}
            {self.listtoString(list(map( lambda node: str(node.nQubits)+self.listtoString(node.loc," "),self.nodes)),nl)}
            {self.listtoString(list(map( lambda x: str(x.n1.id)+str(len(tmp_dict[x])) ,tmp_dict )),nl)}
            """

    def toFullString(self):
        nl="\n"
        tmp_dict=self.grouby_dict(self.links,lambda x:x.n1.to(x.n2))
        return f"""
            {self.n}
            {self.alpha}
            {self.q}
            {self.k}
            {self.listtoString(list(map( lambda node: str(node.remainingQubits)+"/"+str(node.nQubits)+self.listtoString(node.loc," "),self.nodes)),nl)}
            {self.listtoString(list(map( lambda x: str(x.n1.id)+str(len(tmp_dict[x]))+str(len(tmp_dict[x])) ,tmp_dict )),nl)}
            """




    def widhPhase2(self,path):
        tmp1=min(list(map(lambda x: x.remainingQubits/2 ,path[1:-1])))
        p1=tmp1 if tmp1 else sys.maxint
        p2= min( list(map( lambda x: sum( 1 for link in x[0].links if ((link.n1==x[1] | link.n2==x[1]) &(not link.assigned))) ,list(zip(path[:-1],path[1:])))))
        return min(path[0].remainingQubits,path[-1].remainingQubits,p1,p2)

    def widhPhase4(self,path):

        return min( list(map( lambda x: sum( 1 for link in x[0].links if ((link.n1==x[1] | link.n2==x[1]) & link.entangled & link.notSwapped())) ,list(zip(path[:-1],path[1:])))))

    def e(self,path,mul,oldP):
        s=len(path)-1
        P=[0.0 for _ in range(mul+1) ]
        p=[0.0]+list(map( lambda x:math.exp(-alpha*+(x.n1.loc-x.n2.loc)) ,path.edges()))
        start =s
        if sum(oldP)== 0:
            for i in range(0,mul):
                oldP[i]=C_(mul,i)*p[1]**(m)+(1-p[1])**(mul-m)
            start=2
        assert len(oldP)==mul+1

        for k in range(start,s):
            for i in range(0,mul):
                exactlyM=C_(mul,i)*p[k]**(i)*(1-p[k])**(mul-i)
                atLeastM=sum(list(map(  lambda x:  C_(mul,x)*p[k]**(x)*(1-p[k]) **(mul-x)   ,    [l for l in range(i+1,mul)])))+exactlyM
                tmp=0
                for l in range(i+1,mul):
                    tmp+=oldP[l]
                P[i]=oldP[i]*atLeastM+exactlyM*tmp
            for i in range(0,mul)     :
                oldP[m]=P[m]

        assert abs(sum(oldP)-1.0)<0.0001

        return sum(list(map(lambda x:x*oldP[x]  ,[i for i in range(1,mul)])))*self.q**(s-1)

def generate(n,q,k,a,degree):
    alpha=a
    controllingD =math.sqrt(edgeLen*edgeLen/n)
    while len(nodeLocs)<n :
        element=[random.uniform(0, 1)*100 for _ in range(2)]
        














