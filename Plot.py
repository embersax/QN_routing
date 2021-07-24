from QN_routing.configs import referenceSetting, nsdList, topoRange, parseLog, id, nList
from QN_routing.utils.utils import ReducibleLazyEvaluation, flatMap
from itertools import groupby
from string import Template


class Plot:
    def __init__(self):
        self.children = []
        self.nameMapping = {"SL": "SLMP", "Online": "Q-CAST", "Online-R": "Q-CAST\\\\R", "CR": "Q-PASS",
                            "CR-R": "Q-PASS\\\\R", "Greedy_H": "Greedy"}

    def rp2Cdf_nsd(self):

        def getName1(d, n, p, q, k):
            return "a2-rp-len-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList1(nsdList):
            return list(map(lambda it: '{} S-D pairs'.format(it) if it > 1 else '{} S-D pair'.format(it), sorted(nsdList)))

        def getX1(max1):
            return list(range(max1+1))

        def getY1(result1, max1):
            return list(map(lambda l: l+[1.0 for _ in range(max1+1-len(l))] if len(l) < max1+1 else l, result1))

        def getName2(d, n, p, q, k):
            return "a2-rp-wid-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList2(nsdList):
            return list(map(lambda it: '{} S-D pairs'.format(it) if it > 1 else '{} S-D pair'.format(it), sorted(nsdList)))

        def getX2(max2):
            return list(range(max2+1))

        def getY2(result2, max2):
            return list(map(lambda l: l+[1.0 for _ in range(max2+1-len(l))] if len(l) < max2+1 else l, result2))

        def getName3(d, n, p, q, k):
            return "a2-rp-wid-per-mp-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList3(nsdList):
            return list(map(lambda it: '{} S-D pairs'.format(it) if it > 1 else '{} S-D pair'.format(it), sorted(nsdList)))

        def getX3(max3):
            return list(range(max3+1))

        def getY3(result3, max3):
            return list(map(lambda l: l+[1 for _ in range(max3+1-len(l))] if len(l) < max3+1 else l, result3))

        def getName4(d, n, p, q, k):
            return "rp-throughput-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList4(names):
            return list(map(lambda it: ' {} '.format(self.nameMapping[it]) if it in self.nameMapping else '  ', names))

        def getX4(max4):
            return list(range(max4+1))

        def getSubX4(max4):
            x = getX4(max4)
            return list(map(lambda it: it[0], [x[i:i+5] for i in range(0, len(x), 5)]))

        def getY4(result4, max4):
            return list(map(lambda l: l+[1 for _ in range(max4+1-len(l))] if len(l) < max4+1 else l, result4))


        d, n, p, q, k, nsd = referenceSetting
        max1 = 0
        result1 = []

        for idx in range(len(nsdList)):
            nsd = nsdList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange, lambda topoIdx: parseLog('dist/'+id(n, topoIdx, q, k, p, d, nsd, 'Online')+'.txt')), lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: len(it.path)-1, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))), key=lambda it: it[0]):
                max1 = max(max1, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it]/s, list(range(max1+1))))
            result1.append(res)

        temp1 = Template("""
        {
          "markerSize": 0,
          "name": "${getName1}",
          "solutionList": ${getSolutionList1},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "Length of recovery path",
          "yTitle": "CDF",
          "xLimit": [0, ${max1}],
          "yLimit": [0, 1],
          "x": ${getX1},
          "y": ${getY1}
        }""")

        string1 = temp1.substitute(getName1=getName1(d, n, p, q, k), getSolutionList1=getSolutionList1(nsdList), max1=max1, getX1=getX1(max1), getY1=getY1(result1, max1))
        self.children.append(string1)

        max2 = 0
        result2 = []

        for idx in range(len(nsdList)):
            nsd = nsdList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: it.width, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max2 = max(max2, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max2 + 1))))
            result2.append(res)

        temp2 = Template("""
        {
          "markerSize": 0,
          "name": "${getName2}",
          "solutionList": ${getSolutionList2},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "Width of recovery path",
          "yTitle": "CDF",
          "xLimit": [0, ${max2}],
          "yLimit": [0, 1],
          "x": ${getX2},
          "y": ${getY2}
        }""")

        string2 = temp2.substitute(getName2=getName2(d, n, p, q, k), getSolutionList2=getSolutionList2(nsdList),
                                 max2=max2, getX2=getX2(max2), getY2=getY2(result2, max2))
        self.children.append(string2)

        max3 = 0
        result3 = []

        for idx in range(len(nsdList)):
            nsd = nsdList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: it.width, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max3 = max(max3, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max3 + 1))))
            result3.append(res)

        temp3 = Template("""
        {
          "markerSize": 0,
          "name": "${getName3}",
          "solutionList": ${getSolutionList3},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "# recovery paths per major path",
          "yTitle": "CDF",
          "xLimit": [0, ${max3}],
          "yLimit": [0, 1],
          "x": ${getX3},
          "y": ${getY3}
        }""")

        string3 = temp3.substitute(getName3=getName3(d, n, p, q, k), getSolutionList3=getSolutionList3(nsdList),
                                   max3=max3, getX3=getX3(max3), getY3=getY3(result3, max3))
        self.children.append(string3)


        maxx, names, result = 0, ['Online', 'Online-R', 'CR', 'CR-R'], []

        for name in names:
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = map(lambda it: sum(it.majorPaths.succ), flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, name) + '.txt'),
                        ))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                maxx = max(maxx, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(maxx + 1))))
            result.append(res)

        temp4 = Template("""
        {
          "markerSize": 0,
            "name": "${getName4}",
          "solutionList": ${getSolutionList4},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "Throughput (eps)",
          "yTitle": "CDF",
          "xLimit": [0, ${max4}],
          "yLimit": [0, 1],
          "x": ${getX4},
            "xTicks&Labels": {getSubX4}
          "y": ${getY4}
        }""")

        string4 = temp4.substitute(getName4=getName4(d, n, p, q, k), getSolutionList4=getSolutionList4(nsdList),
                                   max4=maxx, getX4=getX4(maxx), getSubX4=getSubX4(maxx), getY3=getY4(result, maxx))
        self.children.append(string4)


    def rp2Cdf_n(self):

        def getName1(d, n, p, q, k):
            return "a2-rp-len-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList1(nList):
            return list(map(lambda it: ' |V| = {} '.format(it), sorted(nList)))

        def getX1(max1):
            return list(range(max1+1))

        def getY1(result1, max1):
            return list(map(lambda l: l+[1.0 for _ in range(max1+1-len(l))] if len(l) < max1+1 else l, result1))


        def getName2(d, n, p, q, k):
            return "a2-rp-wid-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList2(nList):
            return getSolutionList1(nList)

        def getX2(max2):
            return list(range(max2+1))

        def getY2(result2, max2):
            return list(map(lambda l: l+[1.0 for _ in range(max2+1-len(l))] if len(l) < max2+1 else [l[idx] for idx in range(len(l)) if idx <= max2], result2))


        def getName3(d, n, p, q, k):
            return "a2-rp-wid-per-mp-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList3(nList):
            return list(map(lambda it: ' |V| = {} '.format(it), sorted(nList)))

        def getX3(max3):
            return list(range(max3+1))

        def getY3(result3, max3):
            return list(map(lambda l: l+[1 for _ in range(max3+1-len(l))] if len(l) < max3+1 else l, result3))

        d, n, p, q, k, nsd = referenceSetting
        max1 = 0
        result1 = []

        for idx in range(len(nList)):
            nsd = nList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths,
                                           lambda it: map(lambda it: len(it.path) - 1, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max1 = max(max1, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max1 + 1))))
            result1.append(res)

        temp1 = Template("""
        {
          "markerSize": 0,
          "name": "${getName1}",
          "solutionList": ${getSolutionList1},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "Length of recovery path",
          "yTitle": "CDF",
          "xLimit": [0, ${max1}],
          "yLimit": [0, 1],
          "x": ${getX1},
          "y": ${getY1}
        }""")

        string1 = temp1.substitute(getName1=getName1(d, n, p, q, k), getSolutionList1=getSolutionList1(nsdList),
                                   max1=max1, getX1=getX1(max1), getY1=getY1(result1, max1))
        self.children.append(string1)

        max2 = 0
        result2 = []

        for idx in range(len(nList)):
            nsd = nList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: it.width, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max2 = max(max2, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max2 + 1))))
            result2.append(res)

        max2 = max(list(map(lambda it: [it.index(y) for y in it if y > 0.99], result2)))

        temp2 = Template("""
                {
                  "markerSize": 0,
                  "name": "${getName2}",
                  "solutionList": ${getSolutionList2},
                  "legendColumn": 2,
                  "legendFontSize": 10,
                  "legendAutomaticallyReorder": False,
                  "xTitle": "Width of recovery path",
                  "yTitle": "CDF",
                  "xLimit": [0, ${max2}],
                  "yLimit": [0, 1],
                  "x": ${getX2},
                  "y": ${getY2}
                }""")

        string2 = temp2.substitute(getName2=getName2(d, n, p, q, k), getSolutionList2=getSolutionList2(nsdList),
                                   max2=max2, getX2=getX2(max2), getY2=getY2(result2, max2))
        self.children.append(string2)


        max3 = 0
        result3 = []

        for idx in range(len(nList)):
            nsd = nList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: it.width, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max3 = max(max3, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max3 + 1))))
            result3.append(res)

        temp3 = Template("""
        {
          "markerSize": 0,
          "name": "${getName3}",
          "solutionList": ${getSolutionList3},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "# recovery paths per major path",
          "yTitle": "CDF",
          "xLimit": [0, ${max3}],
          "yLimit": [0, 1],
          "x": ${getX3},
          "y": ${getY3}
        }""")

        string3 = temp3.substitute(getName3=getName3(d, n, p, q, k), getSolutionList3=getSolutionList3(nsdList),
                                   max3=max3, getX3=getX3(max3), getY3=getY3(result3, max3))
        self.children.append(string3)



if __name__ == '__main__':
    plot = Plot()
    plot.rp2Cdf_nsd()