

def dynSearch(xMin,xMax,yTarget,f,fIsIncreasing,precision):
    x=(xMin + xMax) / 2
    step=x
    for i in rnage(0,100):
        step/=2
        y=f(x)
        if math.abs(y-yTarget)<math.abs(precision):
            break
        if (y>yTarget) ^ fIsIncreasing:
            s+=step
        else:
            x-=step
    return x