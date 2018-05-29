from __future__ import print_function

import numpy as np
import pandas as pd

from ConvLSTM.utils import string2timestamp


class STMatrix():
    """

    docstring for STMatrix

    """
    def __init__(self, data, date, T1=288, T2=96, T3=48, T4=24, CheckComplete = True):
        super(STMatrix, self).__init__()
        data_01 = data[0]
        data_02 = data[1]
        data_03 = data[2]
        data_04 = data[3]
        date_01 = date[0]
        date_02 = date[1]
        date_03 = date[2]
        date_04 = date[3]
        print(date_01.shape, data_01.shape)
        print(date_02.shape, data_02.shape)
        print(date_03.shape, data_03.shape)
        print(date_04.shape, data_04.shape)
        assert (len(data_01) == len(date_01))
        assert (len(data_02) == len(date_02))
        assert (len(data_03) == len(date_03))
        assert (len(data_04) == len(date_04))
        self.data_01 = data_01
        self.data_02 = data_02
        self.data_03 = data_03
        self.data_04 = data_04

        self.date_01 = date_01
        self.date_02 = date_02
        self.date_03 = date_03
        self.date_04 = date_04

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4

        self.pd_date_01 = string2timestamp(date_01, T=T1)
        self.pd_date_02 = string2timestamp(date_02, T=T2)
        self.pd_date_03 = string2timestamp(date_03, T=T3)
        self.pd_date_04 = string2timestamp(date_04, T=T4)
        if CheckComplete:
            self.check_complete()
        self.make_index()
    def make_index(self):
        self.get_index_01 = dict()
        self.get_index_02 = dict()
        self.get_index_03 = dict()
        self.get_index_04 = dict()
        for i, ts in enumerate(self.pd_date_01):
            self.get_index_01[ts] = i
        for i, ts in enumerate(self.pd_date_02):
            self.get_index_02[ts] = i
        for i, ts in enumerate(self.pd_date_03):
            self.get_index_03[ts] = i
        for i, ts in enumerate(self.pd_date_04):
            self.get_index_04[ts] = i
    def check_complete(self):
        missing_date_01 = []
        missing_date_02 = []
        missing_date_03 = []
        missing_date_04 = []

        offset1 = pd.DateOffset(minutes=24*60//self.T1)
        offset2 = pd.DateOffset(minutes=24*60//self.T2)
        offset3 = pd.DateOffset(minutes=24*60//self.T3)
        offset4 = pd.DateOffset(minutes=24*60//self.T4)

        pd_date1 = self.pd_date_01
        pd_date2 = self.pd_date_02
        pd_date3 = self.pd_date_03
        pd_date4 = self.pd_date_04

        i = 1
        while i < (len(pd_date1)):
            if pd_date1[i-1]+offset1 != pd_date1[i]:
                missing_date_01.append('(%s--%s)' % (pd_date1[i-1], pd_date1[i]))
            i += 1
        for v in missing_date_01:
            print(v)

        i = 1

        while i < (len(pd_date2)):
            if pd_date2[i-1]+offset2 != pd_date2[i]:
                missing_date_02.append('(%s--%s)' % (pd_date2[i-1], pd_date2[i]))
            i += 1
        for v in missing_date_02:
            print(v)

        i = 1

        while i < (len(pd_date3)):
            if pd_date3[i - 1] + offset3 != pd_date3[i]:
                missing_date_03.append('(%s--%s)' % (pd_date3[i - 1], pd_date3[i]))
            i += 1
        for v in missing_date_03:
            print(v)

        i = 1

        while i < (len(pd_date4)):
            if pd_date4[i - 1] + offset4 != pd_date4[i]:
                missing_date_04.append('(%s--%s)' % (pd_date4[i - 1], pd_date4[i]))
            i += 1
        for v in missing_date_04:
            print(v)
        assert len(missing_date_01) == 0
        assert len(missing_date_02) == 0
        assert len(missing_date_03) == 0
        assert len(missing_date_04) == 0
    def get_matrix_01(self, date_01):
        return self.data_01[self.get_index_01[date_01]]
    def get_matrix_02(self, date_02):
        return self.data_02[self.get_index_02[date_02]]
    def get_matrix_03(self,date_03):
        return  self.data_03[self.get_index_03[date_03]]
    def get_matrix_04(self,date_04):
        return self.data_04[self.get_index_04[date_04]]
    def save(self,fname):
        pass
    def check_it_01(self,depends):
        for d in depends:
            if d not in self.get_index_01.keys():
                return False
        return True
    def check_it_02(self,depends):
        for d in depends:
            if d not in self.get_index_02.keys():
                return False
        return True
    def check_it_03(self,depends):
        for d in depends:
            if d not in self.get_index_03.keys():
                return False
        return True
    def check_it_04(self,depends):
        for d in depends:
            if d not in self.get_index_04.keys():
                return False
        return True
    def create_dataset(self,len_closeness=3, len_trend=3,len_period=3, TrendInterval=7, Period_interval=1):
        """

        :param len_closeness:
        :param len_trend:
        :param len_period:
        :param TrendInterval:
        :param Period_interval:
        :return:
        """
        offset_frame_01 = pd.DateOffset(minutes=24*60//self.T1)
        offset_frame_02 = pd.DateOffset(minutes=24*60//self.T2)
        offset_frame_03 = pd.DateOffset(minutes=24*60//self.T3)
        offset_frame_04 = pd.DateOffset(minutes=24*60//self.T4)
        XC = []
        XP = []
        XT = []
        Y_01 = []
        Y_02 = []
        Y_03 = []
        Y_04 = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),
                   [Period_interval*self.T1*j for j in range(1, len_period+1)],
                   [TrendInterval*self.T1*j for j in range(1, len_trend+1)]
                ]
        i = max(self.T1*TrendInterval*len_trend, self.T1*Period_interval*len_period, len_closeness)
        while i < len(self.pd_date_01):
            # print("第%i轮" % i)
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it_01([self.pd_date_01[i]-j*offset_frame_01 for j in depend])
            Flag = Flag and self.check_it_02([self.pd_date_01[i]]) and \
                self.check_it_03([self.pd_date_01[i]]) and self.check_it_04([self.pd_date_01[i]])
            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix_01(self.pd_date_01[i]-j*offset_frame_01)for j in depends[0]]
            x_p = [self.get_matrix_01(self.pd_date_01[i]-j*offset_frame_01)for j in depends[1]]
            x_t = [self.get_matrix_01(self.pd_date_01[i]-j*offset_frame_01)for j in depends[2]]
            y1 = self.get_matrix_01(self.pd_date_01[i])
            y2 = self.get_matrix_02(self.pd_date_01[i])
            y3 = self.get_matrix_03(self.pd_date_01[i])
            y4 = self.get_matrix_04(self.pd_date_01[i])
            if len_closeness > 0:
                XC.append(np.stack(x_c))
            if len_period > 0:
                XP.append(np.stack(x_p))
            if len_trend > 0:
                XT.append(np.stack(x_t))
            Y_01.append(y1)
            Y_02.append(y2)
            Y_03.append(y3)
            Y_04.append(y4)
            timestamps_Y.append(self.date_01[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y_01 = np.asarray(Y_01)
        Y_02 = np.asarray(Y_02)
        Y_03 = np.asarray(Y_03)
        Y_04 = np.asarray(Y_04)
        print('XC shape:', XC.shape, 'XP shape:', XP.shape, 'XT shape:', XT.shape, 'Y_01 shape', Y_01.shape,\
              'Y_02 shape:', Y_02.shape, 'Y_03 shape: ', Y_03.shape, 'Y_04.shape: ', Y_04.shape
              )
        return XC, XP, XT, Y_01, Y_02, Y_03, Y_04, timestamps_Y
if __name__ == '__main__':
    pass






















