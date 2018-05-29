import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
import h5py
import time
from collections import OrderedDict
# coding: utf-8
s = "1 2"
a = s.split(" ")[1]
print(a)

"""
for i in range(1, 2):
    i = "%02d" % i
    for j in range(1, 1441):
        j = "%04d" % j

        index = '201501' + str(i) + str(j)
        d[index] = np.zeros((8, 8), dtype=np.float32)
        d[index][0][0]=100
        print(d[index])

cday = datetime.strptime('2015/01/15 00:01', '%Y/%m/%d %H:%M')
year = str(cday.year)
month = "%02d" % cday.month
month = str(month)
day = "%02d" % cday.day
day = str(day)
print(type(year))
print(year)
key = cday.hour*12+math.floor(cday.minute/5)
key = "%03d" % key
res = year+""+month+""+day+""+str(key)
print(res)

columns = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
           'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude', 'RateCodeID',
           'store_and_fwd_flag', 'dropoff_longitude', 'dropoff_latitude', 'payment_type',
           'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
           'total_amount'
           ]
dir = os.path.join('H:\\NewYork Taxi Data\\yellow_tripdata_2015-02.csv')
print(dir)
data = pd.read_csv(dir, names=columns, index_col=False)
data = data.loc[1:, ['tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude']]
print(data.shape)
for indexes in data.index:
    date = data.loc[indexes, ['tpep_pickup_datetime']][0]
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')


DATAPATH='F:\\DeepST\\'
fname = os.path.join(DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(13))
f = h5py.File(fname, 'r')
data = f['data'].value
timestamps = f['date'].value
s = timestamps[0]
e = timestamps[-1]
print(s, e)
year, month, day = map(int, [s[0:4], s[4:6], s[6:8]])
ts = time.strptime("%04i-%02i-%02i" % (year, month, day), '%Y-%m-%d')
year, month, day = map(int, [e[0:4], e[4:6], e[6:8]])
te = time.strptime("%04i-%02i-%02i" % (year, month, day), '%Y-%m-%d')
print(ts, te)
nb_timeslot = (time.mktime(te)-time.mktime(ts))/1800+48
print(data.shape, nb_timeslot/48)
print(timestamps.shape)

columns = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
              'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude', 'RateCodeID',
               'store_and_fwd_flag', 'dropoff_longitude', 'dropoff_latitude', 'payment_type',
               'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
               'total_amount'
              ]
newdir = os.path.join(dir, 'Feb.csv')
data = pd.DataFrame(pd.read_csv(newdir, names=columns, index_col=False, low_memory=False))
data['pickup_longitude'] = data.loc[1:, ['pickup_longitude']].astype(float)
data['pickup_latitude'] = data.loc[1:,['pickup_latitude']].astype(float)
data = data[data['pickup_longitude']<-70]
data = data[data['pickup_latitude']>35]
data = data[['pickup_longitude', 'pickup_latitude']]
data.to_csv("H:\\NewYork Taxi Data\\data.csv")

print(data.dtypes)

des = 'H:\\NewYork Taxi Data\\test1.h5'
dict = {"One": 1, "Two": 2, "Three": 3}
lable = np.zeros(shape=(3, 3))
lable1 = np.ones(shape=(3, 3))
data = []
data.append(lable)
data.append(lable1)
file = h5py.File("H:\\NewYork Taxi Data\\test1.h5", 'w')
file.create_dataset("data", data=data)
file.close()


dict = {20150125044: 33, 20150117001: 10, 20150112020: 20,20150125043:5}
#dict = {44: 33, 1: 10, 20: 20}
#print(20150117001-20150125049)
d = OrderedDict(sorted(dict.items(), key=lambda d: d[0]))
for key, value in d.items():
    print(key, " ", value)

file = h5py.File("H:\\NewYork Taxi Data\\test1.h5", 'a')
a = np.zeros((2, 2))
file.create_dataset("data_01", data=a)
file.close()
file = h5py.File("H:\\NewYork Taxi Data\\test1.h5", 'a')
b = np.zeros((2, 2))
file.create_dataset("data_02", data=b)
file.close()
a = np.array([[[1, 2, 3], [4, 5, 6]]])
b = np.array([[[2, 2, 2], [2, 2, 2]]])
c = a * b
print(c)
"""



