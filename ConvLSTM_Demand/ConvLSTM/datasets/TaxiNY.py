"""
load NY Data from multiple sources as follows:
          holiday data,
          meteorologic data
"""

from __future__ import print_function

import os
import pickle

import h5py
import numpy as np

from ConvLSTM.preprocessing.minmax_normalization import MinMaxNormalization
from ConvLSTM.preprocessing import timestamp2vec
from . import load_stdata, stat
from .STMatrix import STMatrix
from ..utils import string2timestamp
T1 = 288
T2 = 96
T3 = 48
T4 = 24
len_test_01 = 288*7
len_test_02 = 96*7
len_test_03 = 48*7
len_test_04 = 24*7
len_test = 24*7
DATAPATH = 'H:\\NewYork Taxi Data'  # self-define

def load_holiday(timeslots, fname= os.path.join(DATAPATH, 'NY','NY_holiday.txt')):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    return H[:, np.newaxis]
def load_metrorol(timeslots, fname = os.path.join(DATAPATH, 'climate.h5')):
    """

    :param timeslots:
    :param fname:
    :return:
    """
    f = h5py.File(fname, 'r')
    Timeslot = f['Date'].value
    SkyConditions = f['SkyConditions'].value
    Weather = f['Weather'].value
    f.close()
    Timeslot = string2timestamp(Timeslot, T=48)
    timeslots = string2timestamp(timeslots, T=288)
    M = dict()
    for i, slot in enumerate(Timeslot):
        M[slot] = i
    SC = []  # SkyConditions
    WR = []  # Weather

    for slot in timeslots:
        cur_id = M[slot]
        predicted_id = cur_id-1
        WR.append(Weather[predicted_id])
        SC.append(SkyConditions[predicted_id])
    WR = np.asarray(WR)
    SC = np.asarray(SC)
    # 0-1 scale
    print('shape:', WR.shape, SC.shape)
    # concatenate all these attributes
    merge_data = np.hstack([WR, SC])
    # print('merge shape:', merge_data.shape)
    return merge_data
def load_data( T = 288, nb_flow = 2, len_closeness = 3, len_period = 3, len_trend = 3,
               preprocess_name='preprocessing.pkl', meta_data=True,
               meteorol_data=True
            ):
    assert (len_closeness+len_trend+len_period > 0)
    # load data
    # 1-6
    des = 'H:\\NewYork Taxi Data\\NY_Train_Data.h5'
    stat(des)
    # print(timestamps)
    # remoce a certain day which dose not have 288 timestamps
    data, timestamps = load_stdata(des)  # load data
    # Four types of data: 5 minutes , 15 minutes, 30 minutes, 60 minutes
    data_01 = data[0]
    data_02 = data[1]
    data_03 = data[2]
    data_04 = data[3]

    date_01 = timestamps[0]
    date_02 = timestamps[1]
    date_03 = timestamps[2]
    date_04 = timestamps[3]
    # Until now it did not know yet whether to use remove_incomplete_days

    # data, timestamps = remove_incomplete_days(data, timestamps, T)
    data_01 = data_01[:, np.newaxis]  # add new axis  number of channel is 1
    data_02 = data_02[:, np.newaxis]
    data_03 = data_03[:, np.newaxis]
    data_04 = data_04[:, np.newaxis]
    # In order to normalize the data using training data
    data_train_01 = data_01[:-len_test_01]
    print('train_data shape: ', data_train_01.shape)
    mmn_01 = MinMaxNormalization()
    mmn_01.fit(data_train_01)
    data_01_mmn = mmn_01.transform(data_01)

    data_train_02 = data_02[:-len_test_02]
    print("train_data shape:", data_train_02.shape)
    mmn_02 = MinMaxNormalization()
    mmn_02.fit(data_train_02)
    data_02_mmn = mmn_02.transform(data_02)

    data_train_03 = data_03[:-len_test_03]
    print("train_data shape: ", data_train_03.shape)
    mmn_03 = MinMaxNormalization()
    mmn_03.fit(data_train_03)
    data_03_mmn = mmn_03.transform(data_03)

    data_train_04 = data_04[:-len_test_04]
    print("train_data shape: ", data_train_04.shape)
    mmn_04 = MinMaxNormalization()
    mmn_04.fit(data_train_04)
    data_04_mmn = mmn_04.transform(data_04)

    fpkl = open(preprocess_name, 'wb')
    obj = [mmn_01, mmn_02, mmn_03, mmn_04]
    pickle.dump(obj, fpkl)
    fpkl.close()

    # To obtain the train data
    XC, XP, XT = [], [], []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []

    date_Y = []
    data_all = [data_01_mmn, data_02_mmn, data_03_mmn, data_04_mmn]
    date_all = [date_01, date_02, date_03, date_04]
    st = STMatrix(data_all, date_all, T1, T2, T3, T4, CheckComplete=False)
    _XC, _XP, _XT, _Y1, _Y2, _Y3, _Y4, _date_Y = st.create_dataset(len_closeness=len_closeness,
                                                                  len_period=len_period,
                                                                  len_trend=len_trend
                                                                  )
    XC.append(_XC)
    XP.append(_XP)
    XT.append(_XT)
    Y1.append(_Y1)
    Y2.append(_Y2)
    Y3.append(_Y3)
    Y4.append(_Y4)
    date_Y += _date_Y
    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(date_Y)
        meta_feature.append(time_feature)
    if meteorol_data:
        # load meterol data
        meteorol_feature = load_metrorol(date_Y)
        meta_feature.append(meteorol_feature)
    print("meta_feature_shape: ", np.shape(time_feature), "meteorol_feature_shape: ", np.shape(meteorol_feature))
    print(np.shape(meta_feature[0]), np.shape(meta_feature[1]))
    meta_feature = np.hstack(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(meta_feature.shape) > 1 else 0
    if metadata_dim < 1:
        metadata_dim = 0
    if meta_data and meteorol_data:
        print('time feature:', time_feature.shape, 'meteorol feature: ', meteorol_feature.shape, 'meta feature: ', meta_feature.shape)
    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y1 = np.vstack(Y1)
    Y2 = np.vstack(Y2)
    Y3 = np.vstack(Y3)
    Y4 = np.vstack(Y4)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape:", XT.shape, "Y1 shape: ", Y1.shape,
          "Y2 shape: ", Y2.shape, "Y3 shape: ", Y3.shape, "Y4 shape: ", Y4.shape
          )
    XC_train, XP_train, XT_train, Y1_train, Y2_train, Y3_train, Y4_train = XC[:-len_test], XP[:-len_test], XT[:-len_test],\
                                                                        Y1[:-len_test], Y2[:-len_test], Y3[:-len_test], Y4[:-len_test]
    XC_test, XP_test, XT_test,Y1_test, Y2_test, Y3_test,Y4_test = XC[-len_test:], XP[-len_test:], XT[-len_test:],\
                                                                  Y1[-len_test:], Y2[-len_test:], Y3[-len_test:], Y4[-len_test:]
    timestamps_train, timestamps_test = date_Y[:-len_test], date_Y[-len_test:]
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    Y_train.append(Y1_train)
    Y_test.append(Y1_test)
    Y_train.append(Y2_train)
    Y_test.append(Y2_test)
    Y_train.append(Y3_train)
    Y_test.append(Y3_test)
    Y_train.append(Y4_train)
    Y_test.append(Y4_test)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape: ', XC_train.shape, Y1_train.shape,\
          'test shape: ', XC_test.shape, Y1_test.shape
          )
    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    for _X in X_train:
        print(_X.shape,)
    print()
    for _X in X_test:
        print(_X.shape)
    print()
    return X_train, Y_train, X_test, Y_test, mmn_01, mmn_02, mmn_03, mmn_04, metadata_dim, timestamps_train, timestamps_test













