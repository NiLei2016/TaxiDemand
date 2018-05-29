from __future__ import print_function
import pandas as pd
from datetime import datetime, timedelta
import time
import os
def string2timestamp(strings, T=48):
    timestampes = []
    time_per_slot = 24.0 / T
    num_per_hour = T // 24
    for t in strings:
        t = str(t)
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:11])-1
        timestampes.append(datetime(year, month, day, hour=int(slot * time_per_slot), minute=
                                    (slot % num_per_hour)*int(60*time_per_slot)
                                    ))
    return timestampes
def timestamp2string(timestamps, T = 48):
    #
    num_per_T = T//24
    return ['%s%02i' % (ts.strftime('%Y-%m-%d'), int(1 + ts.to_datetime().hour * num_per_T + ts.to_datetime.minute / (60//num_per_T))) for
            ts in timestamps]



