
from itertools import combinations, groupby
import math
def dynSearch(xMin,xMax,yTarget,f,fIsIncreasing,precision):
    x=(xMin + xMax) / 2
    step=x
    for i in range(0,100):
        step/=2
        y=f(x)
        if math.abs(y-yTarget)<math.abs(precision):
            break
        if (y>yTarget) ^ fIsIncreasing:
            x+=step
        else:
            x-=step
    return x
def groupby_dict(s, f):
    tmp = {}
    for a, b in groupby(s, f):
        for thing in b:
            tmp[a] = b
    return tmp
# here I implemet factorial as two input functions instead of an extended function of Int as Shouqian does
def factorial_(first,until):
    initial=0
    for i in range(until+1,first):
        initial*=i
    return initial

def C_(totalNum,targetNum):
    return factorial_(totalNum,totalNum - targetNum)/factorial_(targetNum,1)
