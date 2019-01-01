# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:24:53 2018

@author: administrator
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
f=open('train_data.csv') 
df=pd.read_csv(f)     #读入股票数据
data=df.iloc[:,3].values   #取第3-10列
data = data[0:430020]
data3 = np.reshape(data,[14334,30])
x = np.mat(np.diff(data3[:,0:10],axis=1))
y = np.mat(np.mean(data3[:,10:30],axis=1) - data3[:,9])
y = y.T
trainData = np.hstack((x,y))
f.close()

f=open('test_data.csv') 
df=pd.read_csv(f)     #读入股票数据
data=df.iloc[:,3].values   #取第3-10列
data3 = np.reshape(data,[1000,10])
testData = np.mat(np.diff(data3[:,0:10],axis=1))
f.close()


rf=RandomForestRegressor()#这里使用了默认的参数设置
rf.fit(x,y)#进行模型的训练

predictions = rf.predict(testData)

import pandas as pd
a = []
b = []
for i in range(1420,10000,10):
    a.append(int(i/10)+1)
    #b.append(np.mean(data[i:i+10]) + predictions[int(i/10)])
    b.append(data[i+9] + predictions[int(i/10)])
#another way to handle
save = pd.DataFrame({'caseid':a,'midprice':b})
save.to_csv('rf.csv',index=False,sep=',')
#  

from sklearn.cross_validation import cross_val_score, ShuffleSplit
X = x
Y = y
names = ['d1','d2','d3','d4','d5','d6','d7','d8','d9']
rf = RandomForestRegressor()
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
print(sorted(scores, reverse=True))