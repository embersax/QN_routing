from configs import referenceSetting, topoRange, parseLog, id, qList, kList, dList, pList, nList, nsdList
import numpy as np
from configs import *
from utils.utils import *
import statistics
class Plot:
    def __init__(self):
        self.chidren=[]
        self.nameMapping={"SL":"SLMP","Online":"Q-CAST","Online-R":"Q-CAST\\\\R", "CR":"Q-PASS","CR-R":"Q-PASS\\\\R","Greedy_H" :"Greedy"}
        self.names=["Online", "SL", "Greedy_H", "CR", "BotCap", "SumDist", "MultiMetric"]
    def plot(self):
        file = open('../plot/last-plot-data.json', mode='w', encoding='utf-8')
        str=f"""
{
  "type": "line",
  "figWidth": 600,
  "figHeight": 350,
  "usetex": false,

  "legendLoc": "best",
  "legendColumn": 1,

  "markerSize": 8,
  "lineWidth": 1,

  "xLog": false,
  "yLog": false,
  "xGrid": false,
  "yGrid": false,

  "xFontSize": 22,
  "xTickRotate": false,
  "yFontSize": 22,
  "legendFontSize": 14,
  "output": true,
  "show": false,

  "children": {self.chidren}
}
"""
        file.write(str)
        file.close()
    def throughputCdf(self):
        nameMapping = {"SL": "SLMP", "Online": "Q-CAST", "Greedy_H": "Greedy"}
        names=["Online", "SL", "Greedy_H", "CR", "BotCap", "SumDist", "MultiMetric"]
        for i in range (1,4):
            d, n, p, q, k, nsd = referenceSetting
            if i ==2:
                p=0.3
            elif i== 3:
                n=400
            max_item = 0
            def f(name):
                s = 0
                sum1 = ReducibleLazyEvaluation(initilizer=lambda x: 0)
                tmp1 = list(flat_map(lambda topoIdx: parseLog( "dist/" +id(n, topoIdx, q, k, p, d, nsd, name)+".txt" ),self,topoRange ))
                tmp2 = list(map( lambda x :sum(x.majorPaths.succ),tmp1))
                tmp3 = groupby_dict(tmp2,lambda x:x)
                tmp4 = list(map(lambda k: (k,len(tmp4[k])), tmp3))
                tmp5 = sorted(tmp4,key=lambda x:x[0])
                for k,v in tmp5:
                    max_item = max(max_item,k)
                    s+=v
                    for i in range(k,1001):
                        sum1[i] = s
                return list(map( lambda  x:sum1[x]/s ,[i for i in range(0,max_item+1)]))

            result = list(map(lambda name: f(name) ,names))
            y_string=[]
            for item in result:
                if len(item)<max_item+1:
                    y_string.append(item+ [1.0]*(max_item+1-len(item)))
                else:
                    y_string.append(item)
            tmp_str = f"""
        {{
          "markerSize": 0,
          "name":  {listtoString(["throughput-cdf-",f"{d}-",f"{n}-",f"{p}",f"{q}-",f"{k}-",f"{nsd}"],"").replace(".","")}
          "solutionList": {  listtoString(list(map( lambda name: f"{nameMapping[name]}",names)))}
          "xTitle": "Throughput (eps)",
          "xTicks&Labels":{  listtoString( list(map(lambda x: x[0],list(divide_chunks([i for i in range(0,max_item+1)],5))) ))},

          "xLimit": [0, {max_item}],
          "yLimit": [0, 1],
          "x": {[i for i in range(0,max_item+1)]}
          "y": {y_string}
        }}
"""
            self.chidren.append(tmp_str)
        def throughPutNsd():
            d, n, p, q, k, nsd = referenceSetting
            def f_Nsd_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,nsdList)))
            def f_Nsd_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,nsdList)))

            def f_Nsd_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,nsdList)))

            result = list(map(lambda name: f_Nsd_1(name), names))
            result2 = list(map(lambda name: f_Nsd_2(name), names))
            result3 = list(map(lambda name: f_Nsd_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-cdf-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "# S-D pairs in one time slot",
                      "yTitle": "Throughput (eps)",,
                      "x": {nsdList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-cdf-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "# S-D pairs in one time slot",
                      "yTitle": "Throughput (eps)",,
                      "x": {nsdList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-cdf-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "# S-D pairs in one time slot",
                      "yTitle": "Throughput (eps)",,
                      "x": {nsdList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])
        def throughPutD():
            d, n, p, q, k, nsd = referenceSetting
            def f_D_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,dList)))
            def f_D_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,dList)))

            def f_D_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,dList)))

            result = list(map(lambda name: f_D_1(name), names))
            result2 = list(map(lambda name: f_D_2(name), names))
            result3 = list(map(lambda name: f_D_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {dList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {dList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {dList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])

        def throughPutP():
            d, n, p, q, k, nsd = referenceSetting

            def f_P_1(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x: sum([path.succ for path in x.majorPaths]),
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd,
                                                                          name) + "txt"), topoRange)))))

                                 , pList)))

            def f_P_2(name):
                return (list(map(lambda nsd:
                                 statistics.mean(
                                     list(map(lambda x: sum([1 for path in x.majorPaths if path.succ > 0]),
                                              # rstlist
                                              list(flat_map(lambda topoIdx: parseLog(
                                                  "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                            topoRange)))))

                                 , pList)))

            def f_P_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x: len(set(list(
                                     map(lambda path: [path[0], path[-1]],
                                         [path for path in x.majorPaths if path.succ > 0])))),
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd,
                                                                          name) + "txt"),
                                                                        topoRange)))))

                                 , pList)))

            result = list(map(lambda name: f_P_1(name), names))
            result2 = list(map(lambda name: f_P_2(name), names))
            result3 = list(map(lambda name: f_P_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {pList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {pList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {pList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1, tmp_str2, tmp_str3])
        def throughPutN():
            d, n, p, q, k, nsd = referenceSetting
            def f_N_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,nList)))
            def f_N_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,nList)))

            def f_N_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,nList)))

            result = list(map(lambda name: f_N_1(name), names))
            result2 = list(map(lambda name: f_N_2(name), names))
            result3 = list(map(lambda name: f_N_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {nList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {nList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {nList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])

        def throughPutQ():
            d, n, p, q, k, nsd = referenceSetting
            def f_Q_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,qList)))
            def f_Q_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,qList)))

            def f_Q_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,qList)))

            result = list(map(lambda name: f_Q_1(name), names))
            result2 = list(map(lambda name: f_Q_2(name), names))
            result3 = list(map(lambda name: f_Q_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {qList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {qList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {qList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])
        def throughPutK():
            d, n, p, q, k, nsd = referenceSetting
            def f_K_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,kList)))
            def f_K_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,kList)))

            def f_K_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,kList)))

            result = list(map(lambda name: f_K_1(name), names))
            result2 = list(map(lambda name: f_K_2(name), names))
            result3 = list(map(lambda name: f_K_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {kList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {kList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {kList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])

    def efficiency(self):
        for mode in range(1, 4):
            if mode == 1:
                names = ["Online", "SL", "Greedy_H", "CR"]
            elif mode == 2:
                names = ["Online", "SL", "Greedy_H"]
            else:
                names = ["Online", "Online-R", "CR", "CR-R"]
            d, n, p, q, k, nsd = referenceSetting
            maxi = 0
            channels = []
            for name in names:
                # TODO: reducible lazy evaluation with just two parameters
                result = {}
                records = []
                for topoIdx in topoRange:
                    records = parseLog("dist/" + id(n, topoIdx, q, k, p, d, nsd, name) + ".txt")
                prov_map = {}
                tmp = []
                for record in records:
                    first = sum(record.majorPaths.succ)
                    second = 0
                    for majorPath in records:
                        second += majorPath.width * (len(majorPath.path) - 1)
                        for recoveryPath in majorPath.recoveryPaths:
                            second += recoveryPath.width * (len(recoveryPath.path) - 1)
                    tmp.append((first, second))
                for pair in tmp:
                    if pair[0] not in prov_map:
                        prov_map[pair[0]] = [pair]
                    else:
                        prov_map[pair[0]].append(pair)
                for pair in prov_map:
                    maxi = max(maxi, pair[0])
                    if result[pair[0]] not in result:
                        result[pair[0]] = [np.mean(float(x) for x in pair[1])]
                    else:
                        result[pair[0]].append(np.mean(float(x) for x in pair[1]))
            # TODO: Check out this par
            for i in range(maxi + 1):
                channels[i] = result[i]
            # TODO: Fix the String
            self.children.append("""
        {
          "name": "${"channels-throughput-$d-$n-$p-$q-$k-$nsd${if (mode == 1) "" else if (mode == 2) "-noA1" else "-rp"}".replace(".", "")}",
          "solutionList": ${names.map { """ "${nameMapping[it] ?: it}" """ }},
          "xTitle": "Throughput (eps)",
          "yTitle": "# occupied channels",
          "x": ${(0..max).toList()},
          "xTicks&Labels": ${(0..max).chunked(5).map { it.first() }},
          "y": ${channels.map { l -> if (l.size < max + 1) l + List(max + 1 - l.size, { Double.NaN }) else l }}
        } """.strip())

    def fairness(self):
        d, n, p, q, k, nsd = referenceSetting
        maxi = 0
        result = []
        for name in self.names:
            s = 0
            # TODO: reducible lazy evaluation with just two parameters
            sum_ = {}
            topoRangeFlatMap = {}
            records = []
            for topoIdx in topoRange:
                records = parseLog("dist/" + id(n, topoIdx, q, k, p, d, nsd, name) + ".txt")
            prov1 = {}
            i = 0
            for record in records:
                majorPath = record.majorPaths
                if (majorPath.path[0], majorPath.path[-1]) not in prov1:
                    prov1[(majorPath.path[0], majorPath.path[-1])] = [majorPath]
                else:
                    prov1[(majorPath.path[0], majorPath.path[-1])].append(majorPath)
                if topoRange[i] not in topoRangeFlatMap:
                    topoRangeFlatMap[topoRange[i]] = [sum(majorPath.width for majorPath in prov1.values())]
                else:
                    topoRangeFlatMap[topoRange[i]].append(sum(majorPath.width for majorPath in prov1.values()))
                i += 1
            final = {}
            for topoIdx in topoRangeFlatMap:
                final[topoIdx] = len(topoRangeFlatMap[topoIdx])
            srtd = sorted(final)
            for k in srtd:
                v = srtd[k]
                maxi = max(maxi, k)
                s += v
                for i in range(k, 1001):
                    sum_[i] = s
            for it in range(0, maxi+1):
                result[it] = sum[it]/float(s)
        self.children.append("""
        {
          "markerSize": 0,
          "name": "${"mp-cdf-$d-$n-$p-$q-$k-$nsd".replace(".", "")}",
          "solutionList": ${names.map { """ "${nameMapping[it] ?: it}" """ }},
          "xTitle": "Total width of allocated major paths",
          "yTitle": "CDF",
          "xLimit": [0, $max],
          "yLimit": [0, 1],
          "x": ${(0..max).toList()},
          "y": ${result.map { l -> if (l.size < max + 1) l + List(max + 1 - l.size, { 1.0 }) else l }}
        }""".strip())

    def recovery1(self):
        q, k, d = qList[0], kList[0], dList[0]
        for p in pList:
            for n in sorted(nList):
                deviation = 0
                if p != pList[0]: deviation += 1
                if n != nList[0]: deviation += 1
                if deviation >= 1: continue
                names = ["CR", "CR-R"]
            results = []
            for name in names:
                for nsd in sorted(nsdList):
                    rList = []
                    for topoIdx in topoRange:
                        rList.append(parseLog("dist/" + id(n, topoIdx, q, k, p, d, nsd, name) + ".txt"))
                    provList = []
                    for record in rList:
                        avg = np.mean(sum(float(majorPath.succ) for majorPath in record.majorPaths))
                        provList.append(avg)
                    results.append(provList)
            results2 = []
            for name in names:
                for nsd in sorted(nsdList):
                    rList = []
                    for topoIdx in topoRange:
                        rList.append(parseLog("dist/" + id(n, topoIdx, q, k, p, d, nsd, name) + ".txt"))
                    provList = []
                    for record in rList:
                        avg = np.mean(len([float(majorPath.succ) for majorPath in record.majorPaths if float(majorPath.succ) > 0]))
                        provList.append(avg)
                    results2.append(provList)
            self.children.append(["""
          {
            "name": "${"a1-rp-throughput-$d-$n-$p-$q-$k-nsd".replace(".", "")}",
            "solutionList": ${names.map { """ "${nameMapping[it] ?: it}" """ }},
            "xTitle": "# S-D pairs in one time slot",
            "yTitle": "Throughput (eps)",
            "x": $nsdList,
            "y": ${results}
          }""".strip(), """
          {
            "name": "${"a1-rp-succ-pairs-$d-$n-$p-$q-$k-nsd".replace(".", "")}",
            "solutionList": ${names.map { """ "${nameMapping[it] ?: it}" """ }},
            "xTitle": "# S-D pairs in one time slot",
            "yTitle": "# succ S-D pairs",
            "x": $nsdList,
            "y": ${results2}
          }""".strip()])





