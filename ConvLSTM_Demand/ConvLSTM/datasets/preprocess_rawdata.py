import numpy as np
import pandas as pd
from datetime import datetime
import math
import h5py
import os
from collections import OrderedDict
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
dict1 = {}
dict2 = {}
dict3 = {}
dict4 = {}
def read_chunk(dir):
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


def preprocess():
    dir = 'H:\\NewYork Taxi Data\\'
    #des = 'H:\\NewYork Taxi Data\\Train_Data_set.h5'
    des = 'H:\\NewYork Taxi Data\\Train_Data.h5'
    margin1 = (LON2 - LON1) / Len
    margin2 = (LAT2 - LAT1) / Len
    cal = {"Mar": 31}
    #  "Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "June": 30
    cal1 = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6}

    columns = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
              'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude', 'RateCodeID',
               'store_and_fwd_flag', 'dropoff_longitude', 'dropoff_latitude', 'payment_type',
               'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
               'total_amount'
              ]
    """
    columns = ["VendorID", "lpep_pickup_datetime", "Lpep_dropoff_datetime", "Store_and_fwd_flag",
               "RateCodeID", "Pickup_longitude", "Pickup_latitude", "Dropoff_longitude", "Dropoff_longitude",
               "Passenger_count", "Trip_distance", "Fare_amount", "Extra", "MTA_tax", "Tip_amount", "Tolls_amount",
               "Ehail_fee", "improvement_surcharge", "Total_amount", "Payment_type", "Trip_type "


    ]
    """

    dict = {}
    for mon, d in cal.items():
        p = cal1[mon]
        p = '%02d' % p  #format month
        newdir = os.path.join(dir, 'yellow_tripdata_2015-'+str(p)+'.csv')
        if os.path.exists(newdir):

            for i in range(1, d+1):

                i = "%02d" % i
                for j in range(1, int((Hour*Minute)/Interval2)+1):

                    j = "%03d" % j
                    w = cal1[mon]
                    m = "%02d" % w
                    index = '2015'+m+i+j
                    index = int(index)
                    dict[index] = np.zeros(shape=(10, 10), dtype=np.float32)

            #data = pd.read_csv(newdir, names=columns, index_col=False)
            data = read_chunk(newdir)
            data = data.loc[1:, ['tpep_pickup_datetime', 'pickup_longitude', 'pickup_latitude']]
            print(data, type(data), data.shape[0])
            for indexes in data.index:
                # print(indexes)
                date = data.loc[indexes, ['tpep_pickup_datetime']][0]

                date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                year = date.year
                month = date.month
                day = date.day
                month = "%02d" % month
                day = "%02d" % day
                hour = date.hour
                minute = date.minute
                no = hour*(60/Interval2)+math.floor(minute/Interval2)+1
                no = "%03d" % no
                key = str(year) + str(month) + str(day) + str(no)
                key = int(key)
                #if key in dict.keys():
                #  print(1)
                lng = data.loc[indexes, ['pickup_longitude']]
                lng = float(lng)
                lat = data.loc[indexes, ['pickup_latitude']]
                lat = float(lat)
                if (lng >= LON1) and (lng <= LON2) and (lat >= LAT1) and (lat <= LAT2):

                    col_index = math.floor((lng - LON1)/margin1)
                    row_index = math.floor((lat - LAT1)/margin2)
                    dict1[key][row_index][col_index] += 1
                    print("indexes:", indexes, "key: ", key, "col_index: ", col_index, "row_index: ", row_index)

    f = h5py.File(des, 'a')
    dict = OrderedDict(sorted(dict1.items(), key=lambda d: d[0]))
    data_all = []
    date_all = []
    for k, v in dict.items():
        if np.amax(v) > 0:
           data_all.append(v)
          # print(np.string_(k))
           date_all.append(np.string_(k))

        #file.create_group()
    #print(date_all)
    #print(data_all)
    f.create_dataset("data_03", data=data_all)
    f.create_dataset("date_03", data=date_all)
    f.close()
preprocess()






