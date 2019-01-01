# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:42:43 2018

@author: administrator
"""

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
'''
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
plt.plot(train['MidPrice'],color='red')
plt.plot(test['MidPrice'],color='blue')

y = train.MidPrice
X = train.drop(['MidPrice'], axis=1).select_dtypes(exclude=['object'])
'''
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.4)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

test_XX = my_imputer.transform(testData)
predictions = my_model.predict(test_XX)

import pandas as pd
a = []
b = []
for i in range(1420,10000,10):
    a.append(int(i/10)+1)
    #b.append(np.mean(data[i:i+10]) + predictions[int(i/10)])
    b.append(data[i+9] + predictions[int(i/10)])
#another way to handle
save = pd.DataFrame({'caseid':a,'midprice':b})
save.to_csv('xgboost.csv',index=False,sep=',')