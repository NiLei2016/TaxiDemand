from __future__ import print_function
import h5py
import time
import datetime

Interval = 5
Hour = 24
Minute = 60
def load_stdata(fname):
    # read data from h5py file
    f = h5py.File(fname, 'r')  # read data depend  h5py key
    data = []
    timestamps = []
    data_01 = f['data_01'].value
    data.append(data_01)  # data_01 :The interval :5 minutes
    timestamps_01 = f['new_date_01'].value  # type of new_date_01 is int64, whereas date_01 is bytes
    timestamps.append(timestamps_01)
    data_02 = f['data_02'].value   # data_02: The interval :15 minutes
    data.append(data_02)
    timestamps_02 = f['new_date_02'].value
    timestamps.append(timestamps_02)
    data_03 = f['data_03'].value  # data_03: The interval :30 minutes
    data.append(data_03)
    timestamps_03 = f['new_date_03'].value
    timestamps.append(timestamps_03)
    data_04 = f['data_04'].value  # data_04: The interval :60 minutes
    data.append(data_04)
    timestamps_04 = f['new_date_04'].value
    timestamps.append(timestamps_04)
    f.close()
    return data, timestamps

def stat(fname):
    def get_slot(f):
        st = f['new_date_01'][0]
        ed = f['new_date_01'][-1]
        st = str(st)
        ed = str(ed)
        year, month, day = map(int, [st[0:4], st[4:6], st[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [ed[0:4], ed[4:6], ed[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        timeslot = (time.mktime(te)-time.mktime(ts))/(Interval * 60) + (Hour * Minute)/Interval
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return timeslot, ts_str, te_str
    with h5py.File(fname, 'r') as f:
        nb_timeslot, st, ed = get_slot(f)
        num = (Hour * Minute)/Interval
        nb_day = int(nb_timeslot/num)
        mmax = f['data_01'].value.max()
        mmin = f['data_01'].value.max()
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape:%s\n' % str(f['data_01'].shape) + \
               '# of days :%i,from %s to %s \n' % (nb_day, st, ed) + \
               '#of timeslots:%i\n' % int(nb_timeslot) + \
               '#of timeslots(available):%i\n' % (f['date_01'].shape[0]) + \
               '#missing ratio of timeslots:%.1f%%\n' % ((1. - float(f['date_01'].shape[0] / nb_timeslot)) * 100) + \
               'max:%.3f,min:%.3f\n' % (mmax, mmin) + \
               '=' * 5 + 'stat' + '=' * 5
        print(stat)




