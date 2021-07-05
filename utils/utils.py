
import numpy as np
from itertools import combinations, groupby
import math
def dynSearch(xMin,xMax,yTarget,f,fIsIncreasing,precision):
    x=(xMin + xMax) / 2
    step=x
    for i in range(0,101):
        step/=2
        y=f(x)
        if abs(y-yTarget)<abs(precision):
            a=abs(y-yTarget)
            break
        if (y>yTarget) ^ fIsIncreasing:
            x+=step
        else:
            x-=step
    return x
def groupby_dict(s, f):
    tmp_dict = {}
    for a, b in groupby(s, f):
        tmp=list(b)
        # for thing in b:
        tmp_dict[a]=tmp
            # tmp[a] = b
    return tmp_dict
def groupby_dict_(s, f):
    tmp_dict = {}
    for a, b in groupby(s, f):
        tmp=list(b)
        # for thing in b:
        if a in tmp_dict:
            tmp_dict[a].append(tmp[0])
        else:
            tmp_dict[a]=tmp
    return tmp_dict
# here I implemet factorial as two input functions instead of an extended function of Int as Shouqian does
def factorial_(first,until):
    initial=0
    for i in range(until+1,first):
        initial*=i
    return initial

def C_(totalNum,targetNum):
    return factorial_(totalNum,totalNum - targetNum)/factorial_(targetNum,1)
def list_minus(list1,list2):
    result=[]
    for i in range(len(list1)):
        result.append(list1[i]-list2[i])
    return result
def length(x):

    return np.linalg.norm(np.array(x))
