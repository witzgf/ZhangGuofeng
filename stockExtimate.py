# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:24:35 2018

@author: administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
plt.plot(train['MidPrice'],color='blue')
plt.title('MidPrice of Train Dataset')
plt.ylabel('MidPrice')
plt.show()
plt.plot(train['LastPrice']*100,color = 'blue')
plt.plot((train['LastPrice']-train['MidPrice']),color = 'blue')

plt.plot(test['MidPrice'],color='blue')
plt.title('MidPrice of Test Dataset')
plt.ylabel('MidPrice')
plt.show()


y = train.MidPrice
X = train.drop(['MidPrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

y = test.MidPrice
X = test.drop(['MidPrice'], axis=1).select_dtypes(exclude=['object'])
test_XX = my_imputer.transform(X.as_matrix())
predictions = my_model.predict(test_XX)


diffs = []
ratio = [] 
ty = train.MidPrice
for i in range(0,430039,30):
    y1 = ty[i+9]
    y2 = np.mean(ty[i+10:i+20])
    diff = y2 - y1
    diffs.append(diff)
    ratio.append(diff/y1)
pd = list(filter(lambda i:i > 0 and i <0.004,diffs))
pd = list(filter(lambda i:i > 0,diffs))
nd = list(filter(lambda i:i >-0.0002 and i <0,diffs))
nd = list(filter(lambda i:i <0,diffs))
pd_mean = np.mean(pd)
pd_median = np.median(pd)
nd_mean = np.mean(nd)
nd_median = np.median(nd)

import pandas as pd
a = []
b = []
c = []
d = []
for i in range(10,10001,10):
    a.append(int(i/10))
    b.append(predictions[i-1])
    if i < 10000:
#        c.append((y[i-1]+y[i])/2)
        minValue = min(y[i-1],y[i])
        maxValue = max(y[i-1],y[i])
        #c.append(np.mean(np.random.uniform(minValue,maxValue,40)))
        #c.append(y[i-1]+(y[i]-y[i-1])*0.1)
        if (y[i] - y[i-1]) > 0:
            c.append(y[i-1]+ pd_median)
        elif (y[i] - y[i-1]) < 0:
            c.append(y[i-1]+ nd_median)
        else:
            c.append(y[i-1]+ 0)
        d.append(y[i]-y[i-1])
    else:
        c.append(y[i-1])
        d.append(y[i-1]-y[i-2])

caseid_column = pd.Series(a, name='caseid')
midprice_column = pd.Series(b, name='midprice')
submision = pd.concat([caseid_column, midprice_column], axis=1)

#another way to handle
save = pd.DataFrame({'caseid':a,'midprice':b})
save = pd.DataFrame({'caseid':a,'midprice':c})
save.to_csv('b8.csv',index=False,sep=',')


'''
import csv    
with open('aa.csv','w') as fout:
        fieldnames = ['caseid','midprice']
        writer = csv.DictWriter(fout, fieldnames = fieldnames)
        writer.writeheader()
        for i in range(len(a)):
            writer.writerow({'caseid':str(a[i]),'midprice':float(b[i])})
            '''