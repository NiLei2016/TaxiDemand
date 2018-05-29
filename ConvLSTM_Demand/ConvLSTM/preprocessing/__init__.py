import numpy as np
import pandas as pd
import datetime
def timestamp2vec(timestamps):
    # tm_weekday range[0,6], Monday is 0
    vec = []
    for t in timestamps:
        p = str(t)
        date = datetime.datetime.strptime(str(p[:8]), '%Y%m%d').weekday()
        vec.append(date)
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)
        # print('v shape', np.shape(v))
        ret.append(v)
    return np.asarray(ret)

def remove_incomplete_days(data, timestamps, T = 288):
    # remove a certain day which has not 48 timestamps
    days = []  # available days
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif (i+T-1) < len(timestamps) and int(timestamps[i+T-1][8:] == T):
            i += T
            days.append(timestamps[i][:8])
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print('incomplete days：', days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
             idx.append(i)
    data = data[idx]
    timestamps = timestamps[idx]
    return data, timestamps



