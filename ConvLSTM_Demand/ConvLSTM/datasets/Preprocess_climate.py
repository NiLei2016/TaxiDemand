import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import math
import h5py
Hour = 24
Minute = 60
def create_dict(Interval, shape):
    Cal = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30}
    dict = {}
    for mon in range(1, 7):
        d = Cal[mon]
        for i in range(1, d + 1):

            i = "%02d" % i
            for j in range(1, int((Hour * Minute) / Interval) + 1):
                j = "%03d" % j
                m = "%02d" % mon
                index = '2015' + m + i + j
                index = int(index)
                dict[index] = np.zeros(shape=shape, dtype=np.float32)
    return dict


def read_chunk(dir):  # data block reading
    reader = pd.read_csv(dir, iterator=True, low_memory=False)
    loop = True
    chunk_size = 10000
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
def process_sky_conditions(sky_conditions):
    res = np.zeros(shape=(6,), dtype=np.float32)
    mark = sky_conditions.split(':', -1)[0]
    if(mark == 'CLR'):
        res[0] = 1
    elif(mark == 'BKN'):
        res[1] = 1
    elif(mark == 'OVC'):
        res[2] = 1
    elif(mark == 'FEW') or (mark == 'SCT'):
        res[3] = 1
    elif(mark == 'VV'):
        res[4] = 1
    elif(mark == '10') or (mark == 'UN'):
        res[5] = 1
    return res
def process_visibility(visibility):
    res = np.zeros(shape=(1,), dtype=np.float32)
    if(visibility[-1]=='V'):
        visibility = visibility[:len(visibility)-1]
    mark = float(visibility)
    res[0] = mark
    return res

def process_weather_type(weather):
    res = np.zeros(shape=(9,), dtype=np.float32)
    key = weather.split("|", -1)[0].split(':', -1)[0]
    if(key=='TS'):
        res[1] = 1
    elif(key=='HZ') or (key == 'BR'):
        res[2] = 1
    elif(key =='-SN'):
        res[3] = 1
    elif(key=='SN'):
        res[4] = 1
    elif(key=='+SN'):
        res[5] = 1
    elif(key=='-RA'):
        res[6] = 1
    elif(key=='RA'):
        res[7] = 1
    elif(key=='+RA'):
        res[8] = 1
    else:
        res[0] = 1
    return res
def Write_To_File(des, dict, name1):
    f = h5py.File(des, 'a')
    data_all = []
    for k, v in dict.items():
        data_all.append(v)
    f.create_dataset(name=name1, data=data_all)
    f.close()
def process():
    #sky conditions
    Interval  = 30
    sky_con = create_dict(Interval,6)
    sky_con1 = {}
    # visibility
    visib = create_dict(Interval,1)
    visib1 = {}
    # Weather type
    Weather = create_dict(Interval,9)
    Weather1 = {}
    dir = "H:\\NewYork Taxi Data\\NY climate.csv"
    des = 'H:\\NewYork Taxi Data\\climate.h5'
    data = read_chunk(dir)
    data = data.loc[1:, ['DATE', 'HOURLYSKYCONDITIONS', 'HOURLYVISIBILITY', 'HOURLYPRSENTWEATHERTYPE']]
    data.fillna('UN')
    for indexes in data.index:
        sky_conditions = data.loc[indexes, ['HOURLYSKYCONDITIONS']]['HOURLYSKYCONDITIONS']
        visibility = data.loc[indexes, ['HOURLYVISIBILITY']]['HOURLYVISIBILITY']
        weather_type = data.loc[indexes, ['HOURLYPRSENTWEATHERTYPE']]['HOURLYPRSENTWEATHERTYPE']
        sky_conditions = str(sky_conditions)
        visibility = str(visibility)
        weather_type = str(weather_type)
        print(type(sky_conditions), sky_conditions, type(visibility))
        if (sky_conditions == 'UN' and visibility == 'UN' and weather_type == 'UN'):
            continue
        date = data.loc[indexes, ['DATE']][0]
        date = datetime.strptime(date, '%Y-%m-%d %H:%M')
        year = date.year
        month = date.month
        day = date.day
        month = "%02d" % month
        day = "%02d" % day
        hour = date.hour
        minute = date.minute
        key1 = pd.Timestamp(datetime(int(year), int(month), int(day), int(hour), minute=math.floor(minute / Interval) * Interval))
        no_01 = hour * (60 / Interval) + math.floor(minute / Interval) + 1
        no_01 = "%03d" % no_01
        key_01 = str(year) + str(month) + str(day) + str(no_01)
        key_01 = int(key_01)

        if(sky_conditions == 'UN'):
            li = np.zeros(shape=(6,), dtype=np.float32)
            li[0] = 1
            sky_con[key_01] = li
            sky_con1[key1] = li
        if(visibility == 'UN'):
            li = np.zeros(shape=(1,), dtype=np.float32)
            li[0] = 1
            visib[key_01]=li
            visib1[key1] = li
        if(weather_type == 'UN'):
            li = np.zeros(shape=(9,), dtype = np.float32)
            li[0] = 1
            Weather[key_01] = li
            Weather1[key1] = li
        print(sky_con[key_01])
        sky_con[key_01] = process_sky_conditions(sky_conditions)
        sky_con1[key1] = process_sky_conditions(sky_conditions)
        visib[key_01] = process_visibility(visibility)
        visib1[key1] = process_visibility(visibility)
        Weather[key_01] = process_weather_type(weather_type)
        Weather1[key1] = process_weather_type(weather_type)
    sky_con_01 = OrderedDict(sorted(sky_con.items(), key=lambda d: d[0]))
    visib_01 = OrderedDict(sorted(visib.items(), key=lambda d: d[0]))
    Weather_01 = OrderedDict(sorted(Weather.items(), key=lambda d: d[0]))
    for (d, x) in sky_con_01.items():
        if(np.max(x) == 0):
            d = str(d)
            y, m, day, slot = int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[8:11])-1

            timestamp = pd.Timestamp(datetime(y, m, day, hour=int(slot * 24.0//48),
                                              minute=(slot % (48//24)*int(60*24.0//48))
                                              ))
            offset = pd.DateOffset(minutes=24*60//48)
            timestamp_01 = timestamp-offset  # last 30 minutes
            timestamp_02 = timestamp+offset  # next 30 minutes
            d = int(d)
            if timestamp_01 in sky_con1.keys():
                p = sky_con1[timestamp_01]
                if(np.max(p)) > 0:

                    sky_con_01[d] = p
                else:
                    sky_con_01[d][0] = 1
            else:
                sky_con_01[d][0] = 1
    for (d, x) in visib_01.items():
        if (np.max(x) == 0):
            d = str(d)
            y, m, day, slot = int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[8:11]) - 1
            timestamp = pd.Timestamp(datetime(y, m, day, hour=int(slot * 24.0 // 48),
                                              minute=(slot % (48 // 24) * int(60 * 24.0 // 48))
                                              ))
            offset = pd.DateOffset(minutes=24 * 60 // 48)
            timestamp_01 = timestamp - offset  # last 30 minutes
            timestamp_02 = timestamp + offset  # next 30 minutes
            d = int(d)
            if timestamp_01 in visib1.keys():
                p = visib1[timestamp_01]
                if (np.max(p)) > 0:

                    visib_01[d] = p
                else:
                    visib_01[d][0] = 1
            else:
                visib_01[d][0] = 1
    for (d, x) in Weather_01.items():
        if (np.max(x) == 0):
            d = str(d)
            y, m, day, slot = int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[8:11]) - 1
            timestamp = pd.Timestamp(datetime(y, m, day, hour=int(slot * 24.0 // 48),
                                              minute=(slot % (48 // 24) * int(60 * 24.0 // 48))
                                              ))
            offset = pd.DateOffset(minutes=24 * 60 // 48)
            timestamp_01 = timestamp - offset  # last 30 minutes
            timestamp_02 = timestamp + offset  # next 30 minutes
            d = int(d)
            if timestamp_01 in Weather1.keys():
                p = Weather1[timestamp_01]
                if (np.max(p)) > 0:

                    Weather_01[d] = p
                else:
                    Weather_01[d][0] = 1
            else:
                Weather_01[d][0] = 1
    dict = [sky_con_01, visib_01, Weather_01]
    name = ["SkyConditions", "Visibility", "Weather"]
    for d, n in zip(dict, name):
        print(n)
        Write_To_File(des, d, n)

    timeslot = []
    for d in sky_con_01.keys():
        timeslot.append(d)
    f = h5py.File(des, 'a')
    f.create_dataset(name='Date', data=timeslot)
    f.close()

if __name__ == "__main__":
    process()