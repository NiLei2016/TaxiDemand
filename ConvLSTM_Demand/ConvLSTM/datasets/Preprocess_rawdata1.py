import numpy as np
import pandas as pd
from datetime import datetime
import math
import h5py
import os
from collections import OrderedDict
import csv
LAT1 = 40.7164
LON1 = -74.0127
LAT2 = 40.7647
LON2 = -73.9685
Len = 10
Interval1 = 5
Interval2 = 15
Interval3 = 30
Interval4 = 60

Hour = 24
Minute = 60

def read_chunk(dir):  # data block reading
    reader = pd.read_csv(dir, iterator=True)
    loop = True
    chunk_size = 1000000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=False)
    return df

def create_dict(Interval,t,mon):
    dict = {}
    for i in range(1, t + 1):

        i = "%02d" % t
        for j in range(1, int((Hour * Minute) / Interval) + 1):
            j = "%03d" % j
            m = "%02d" % mon
            index = '2015' + m + i + j
            index = int(index)
            dict[index] = np.zeros(shape=(10, 10), dtype=np.float32)
    return dict
def create_dict1(Interval,t,mon):
    dict = {}
    for i in range(1, t + 1):

        i = "%02d" % i
        for j in range(1, int((Hour * Minute) / Interval) + 1):
            j = "%03d" % j
            m = "%02d" % mon
            index = '2015' + m + i + j
            index = int(index)
            dict[index] = 0
    return dict

def Write_To_File(des, dict, name1, name2):
    f = h5py.File(des, 'a')
    data_all = []
    date_all = []
    for k, v in dict.items():
        if np.amax(v) > 0:
            data_all.append(v)
            date_all.append(np.string_(k))

    f.create_dataset(name1, data=data_all)
    f.create_dataset(name2, data=date_all)
    f.close()

def count_order():
    dir = 'H:\\NewYork Taxi Data\\'
    des = 'H:\\NewYork Taxi Data\\Total_order_per_month_June.csv'
    des2 = 'H:\\NewYork Taxi Data\\Total_order_per_hour.csv'
    csvfile = open(des, 'w', newline="")
    csvfile2 = open(des2, 'w', newline="")
    writer = csv.writer(csvfile)
    writer2 = csv.writer(csvfile2)
    #writer.writerow(['time', 'order'])
    t = 30
    mon = 6
    dict = {}
    dict2 = create_dict1(Interval4, t, mon)
    #print(dict2)
    m = "%02d" % mon
    for i in range(1, t + 1):
        i = "%02d" % i
        index = '2015' + m + i
        index = int(index)
        dict[index] = 0

    newdir = os.path.join(dir, 'yellow_tripdata_2015-' + str(m) + '.csv')
    data = read_chunk(newdir)
    data = data.loc[1:, ['tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude']]
    for indexes in data.index:
        date = data.loc[indexes, ['tpep_pickup_datetime']][0]
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        year = date.year
        month = date.month
        day = date.day
        month = "%02d" % month
        day = "%02d" % day
        hour = date.hour
        minute = date.minute
        no_04 = hour * (60 / Interval4) + math.floor(minute / Interval4) + 1
        no_04 = "%03d" % no_04
        key_04 = str(year) + str(month) + str(day) + str(no_04)
        key_04 = int(key_04)
        key_01 = str(year) + str(month) + str(day)
        key_01 = int(key_01)
        lng = data.loc[indexes, ['pickup_longitude']]
        lng = float(lng)
        lat = data.loc[indexes, ['pickup_latitude']]
        lat = float(lat)
        if (lng >= LON1) and (lng <= LON2) and (lat >= LAT1) and (lat <= LAT2):
            dict[key_01] += 1
            dict2[key_04] += 1
            print(indexes, key_01, key_04)
    dict_01 = OrderedDict(sorted(dict.items(), key=lambda d: d[0]))
    dict_02 = OrderedDict(sorted(dict2.items(), key=lambda d: d[0]))
    no = ["date", "order"]
    writer.writerow(no)
    writer2.writerow(no)
    for (key, d) in dict_01.items():
        p = (key, d)
        writer.writerow(p)
    csvfile.close()
    for(key, d) in dict_02.items():
        p = (key, d)
        writer2.writerow(p)
    csvfile2.close()




def preprocess():
    dir = 'H:\\NewYork Taxi Data\\'
    des1 = 'H:\\NewYork Taxi Data\\Train_Data1.h5'
    des2 = 'H:\\NewYork Taxi Data\\Train_Data.h5'
    des3 = 'H:\\NewYork Taxi Data\\Train_Data3.h5'
    des4 = 'H:\\NewYork Taxi Data\\Train_Data4.h5'
    #des = [des1, des2, des3, des4]
    des = [des1, des3, des4]
    margin1 = (LON2 - LON1) / Len
    margin2 = (LAT2 - LAT1) / Len
    t = 31
    mon = 3
    dict1 = create_dict(Interval1, t, mon)
    #dict2 = create_dict(Interval2, t, mon)
    dict3 = create_dict(Interval3, t, mon)
    dict4 = create_dict(Interval4, t, mon)
    columns = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
              'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude', 'RateCodeID',
               'store_and_fwd_flag', 'dropoff_longitude', 'dropoff_latitude', 'payment_type',
               'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
               'total_amount'
              ]

    p = "%02d" % mon
    newdir = os.path.join(dir, 'yellow_tripdata_2015-'+str(p)+'.csv')
    data = read_chunk(newdir)
    data = data.loc[1:, ['tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude']]
    print(data, type(data), data.shape[0])
    for indexes in data.index:
        date = data.loc[indexes, ['tpep_pickup_datetime']][0]
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        year = date.year
        month = date.month
        day = date.day
        month = "%02d" % month
        day = "%02d" % day
        hour = date.hour
        minute = date.minute

        no_01 = hour*(60/Interval1)+math.floor(minute/Interval1)+1
        no_01 = "%03d" % no_01
        key_01 = str(year) + str(month) + str(day) + str(no_01)
        key_01 = int(key_01)
        """
        no_02 = hour*(60/Interval2)+math.floor(minute/Interval2)+1
        no_02 = "%03d" % no_02
        key_02 = str(year)+str(month)+str(day)+str(no_02)
        key_02 = int(key_02)
        """
        no_03 = hour * (60 / Interval3) + math.floor(minute / Interval3) + 1
        no_03 = "%03d" % no_03
        key_03 = str(year) + str(month) + str(day) + str(no_03)
        key_03 = int(key_03)

        no_04 = hour * (60 / Interval4) + math.floor(minute / Interval4) + 1
        no_04 = "%03d" % no_04
        key_04 = str(year) + str(month) + str(day) + str(no_04)
        key_04 = int(key_04)

        lng = data.loc[indexes, ['pickup_longitude']]
        lng = float(lng)
        lat = data.loc[indexes, ['pickup_latitude']]
        lat = float(lat)
        if (lng >= LON1) and (lng <= LON2) and (lat >= LAT1) and (lat <= LAT2):
            col_index = math.floor((lng - LON1)/margin1)
            row_index = math.floor((lat - LAT1)/margin2)
            dict1[key_01][row_index][col_index] += 1
            #dict2[key_02][row_index][col_index] += 1
            dict3[key_03][row_index][col_index] += 1
            dict4[key_04][row_index][col_index] += 1
            print("indexes:", indexes, "key: ", key_01, "col_index: ", col_index, "row_index: ", row_index)

    dict_01 = OrderedDict(sorted(dict1.items(), key=lambda d: d[0]))
    #dict_02 = OrderedDict(sorted(dict2.items(), key=lambda d: d[0]))
    dict_03 = OrderedDict(sorted(dict3.items(), key=lambda d: d[0]))
    dict_04 = OrderedDict(sorted(dict4.items(), key=lambda d: d[0]))

    #dict = [dict_01, dict_02, dict_03, dict_04]
    dict = [dict_01, dict_03, dict_04]
    name1 = "data_03"
    name2 = "date_03"

    for di, d in zip(des, dict):
        Write_To_File(di, d, name1, name2)  # write data to HDF file
def concat_data(dir):
    f = h5py.File(dir, 'r')
    data_01 = f['data_01'].value
    data_02 = f['data_02'].value
    data_03 = f['data_03'].value
    data_04 = f['data_04'].value
    data_05 = f['data_05'].value
    data_06 = f['data_06'].value
    data = np.vstack((data_01, data_02, data_03, data_04, data_05, data_06))
    date_01 = f['date_01'].value
    date_02 = f['date_02'].value
    date_03 = f['date_03'].value
    date_04 = f['date_04'].value
    date_05 = f['date_05'].value
    date_06 = f['date_06'].value
    date = []
    date.extend(date_01)
    date.extend(date_02)
    date.extend(date_03)
    date.extend(date_04)
    date.extend(date_05)
    date.extend(date_06)
    #date = np.vstack([date_01, date_02, date_03, date_04, date_05, date_06])
    print("data shape: ", np.shape(data), "date shape: ", np.shape(date))
    return data, date

def concat():
    dir1 = 'H:\\NewYork Taxi Data\\Train_Data1.h5'
    dir2 = 'H:\\NewYork Taxi Data\\Train_Data.h5'
    dir3 = 'H:\\NewYork Taxi Data\\Train_Data3.h5'
    dir4 = 'H:\\NewYork Taxi Data\\Train_Data4.h5'
    des = 'H:\\NewYork Taxi Data\\NY_Train_Data.h5'  #type of the date is bytes, so need to transform date into int or string
    data_01, date_01 = concat_data(dir1)
    data_02, date_02 = concat_data(dir2)
    data_03, date_03 = concat_data(dir3)
    data_04, date_04 = concat_data(dir4)
    data_all = [data_01, data_02, data_03, data_04]
    date_all = [date_01, date_02, date_03, date_04]
    name1 = ['data_01', 'data_02', 'data_03', 'data_04']
    name2 = ['date_01', 'date_02', 'date_03', 'date_04']
    for (data, date, name1, name2) in zip(data_all, date_all,name1,name2):
        f = h5py.File(des, 'a')
        f.create_dataset(name=name1, data=data)
        f.create_dataset(name=name2, data=date)
    f.close()

if __name__ == "__main__":
   # preprocess()
   #concat()
   count_order()
