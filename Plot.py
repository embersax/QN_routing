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
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))                          ,

                        nsdList)))
            def f_Nsd_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))                          ,

                        nsdList)))

            def f_Nsd_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange))))),

                                 nsdList)))

            result = list(map(lambda name: f_Nsd_1(name), names))
            result2 = list(map(lambda name: f_Nsd_2(name), names))
            result3 = list(map(lambda name: f_Nsd_3(name), names))




