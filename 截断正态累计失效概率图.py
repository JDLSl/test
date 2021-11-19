# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:38:59 2021

@author: Dell
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import norm

def normcdf(x,mu,sigma):
    return norm.cdf((x-mu)/sigma)

def truncnormcdf(x,mu,sigma):    
    myclip_a=0
    myclip_b=float("Inf")
    def get_ab(my_mean,my_std):
        a, b =  (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        return a,b
    lower, upper = get_ab(mu,sigma)
    z=stats.truncnorm.cdf(x,lower, upper, loc=mu, scale=sigma)
    return z

def weibullcdf(x,scale,shape):
    return 1-np.exp(-(x/scale)**shape)

prob = pd.DataFrame(columns=["z","zmin","zmax"])
endtime=105#确定横坐标终点 需根据实际修改
x=np.linspace(0,endtime-1,endtime)#横坐标均匀分布
#norm 1为下限，2为上限
#16不截断
#mu=121.238
#sigma=51.930
#mu1=117.388
#sigma1=48.650
#mu2=126.561  
#sigma2=55.286
#16截断
#mu=84.684
#sigma=61.333
#mu1=79.246
#sigma1=58.128
#mu2=89.789 
#sigma2=65.041
#17不截断
mu=34.19
sigma=20.63
mu1=33.58
sigma1=20.09
mu2=34.93
sigma2=21.27

#17截断
#mu=72.956
#sigma=59.253
#mu1=68.501
#sigma1=56.473
#mu2=78.046
#sigma2=62.855

z1=truncnormcdf(x,mu,sigma)#参数估计值
prob["z"] = z1
z2=truncnormcdf(x,mu1,sigma1)
z3=truncnormcdf(x,mu2,sigma1)
z4=truncnormcdf(x,mu1,sigma2)
z5=truncnormcdf(x,mu2,sigma2)
z=[]
zmin=[]
zmax=[]
for i in range(endtime):
    z.append(z1[i])
    zmin.append(min(z2[i],z3[i],z4[i],z5[i]))
    zmax.append(max(z2[i],z3[i],z4[i],z5[i]))

data=pd.DataFrame()
data['z']=z
data['zmin']=zmin
data['zmax']=zmax

#data.to_excel("D:/研究生/项目/神华/钩缓/牵引杆/C80/截断正态保守累计概率.xlsx")

plt.plot(x,z1,'b-',linewidth=1.2)

font_song = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=12)

plt.xlabel('运行里程（wkm）',fontproperties=font_song)
plt.ylabel('累积失效概率',fontproperties=font_song)
plt.xticks(fontsize = 13,alpha=2.0)
plt.yticks(fontsize = 13)

plt.fill_between(x,zmin,zmax,alpha=0.3,facecolor='b')

plt.grid(True)

plt.show()

