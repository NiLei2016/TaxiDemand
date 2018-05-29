from __future__ import print_function

import os
import pickle
import time

import h5py
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from ConvLSTM.utils.metrics import  rmse
from ConvLSTM.datasets import TaxiNY
from ConvLSTM.models.ConvLstm import seqConvLSTM_CPTM, seqLSTM_CPTM, seqGRU_CPTM, seqConv_CPTM,seqCNN_LSTM_CPTM
from ConvLSTM.utils import metrics
from ConvLSTM.models.STResNet import stresnet

np.random.seed(1337)  # for reproducibility
import tensorflow as tf
DATAPATH ='H:\\NewYork Taxi Data\\'  # data path
CACHEDATA = True
path_cache = os.path.join(DATAPATH, 'CACHE')
nb_epoch = 500  # number of epoch at training
nb_epoch_cont = 100   # number of epoch at training stage
batch_size = 24  # batch size
T = 24  # number of time intervals at a day
lr = 0.0002  # learning rate
len_closeness = 13  # length of closeness dependent sequence
len_period = 1  # length of period dependent sequence
len_trend = 1  # length of trend dependent sequence
nb_conlstm_unit = 12  # number of convlstm units

nb_flow = 1  # channel of data
# split data into two subsets : Train & Test, of which the set is the last one weeks

days_test = 7
len_test = T * days_test
map_height, map_width = 10, 10  # grid size
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) and os.path.exists(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)
def build_model(external_dim):
    c_conf = (len_closeness, nb_flow, map_height, map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height, map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height, map_width) if len_trend > 0 else None
    # with K.get_session():
    #model = seqConvLSTM_CPTM(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim)
    model = seqCNN_LSTM_CPTM(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim)
    #model = seqConvLSTM_CPTM(c_conf=c_conf, p_conf=p_conf, external_dim=external_dim)
    #model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=12)
    adam = Adam(lr)
    model.compile(loss='mse', optimizer=adam, loss_weights=[1., 1., 1., 1.],
                  metrics=[metrics.rmse])
    model.summary()
    # K.clear_session()
    # from keras.utils.visualize until import plot
    # plot(model, to_file='model.png',show_shapes=True)
    return model
def read_cache(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))
    mmn_01 = mmn[0]
    mmn_02 = mmn[1]
    mmn_03 = mmn[2]
    mmn_04 = mmn[3]
    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []

    for i in range(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
        Y_train.append(f['Y_train_%i' % i].value)
        Y_test.append(f['Y_test_%i' % i].value)

    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()
    return X_train, Y_train, X_test, Y_test, mmn_01, mmn_02, mmn_03, mmn_04, external_dim, timestamp_train, timestamp_test
def Write_To_File(dir, arr):
    f = open(dir, 'a')
    for k, data in enumerate(arr):
        f.writelines(str(k+1)+" "+str(round(data))+'\n')
    f.close()
def Write_To_File2(dir, arr):
    f = open(dir, 'a')
    for k, data in enumerate(arr):
        f.write(str(k+1)+" ")
        for num in data:
            f.write(str(round(num))+" ")
        f.write("\n")
    f.close()
def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))
    print("X_train shape: ", len(X_train), X_train[0].shape, "Y_train shape: ", len(Y_train), "X_test shape", len(X_test),
          "Y_test shape: ", len(Y_test), Y_test[0].shape
          )
    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    for i, data in enumerate(Y_train):
        h5.create_dataset('Y_train_%i' % i, data=data)
    for i, data in enumerate(Y_test):
        h5.create_dataset('Y_test_%i' % i, data=data)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()
def main():
    # load data
    print("loading data.....")
    ts = time.time()
    fname = os.path.join(DATAPATH, 'CACHE', 'TaxiNY_C{}_P{}_T{}_noExternal.h5'.format(len_closeness, len_period, len_trend))
    if os.path.exists(fname) and CACHEDATA:
        X_train, Y_train, X_test, Y_test, mmn1, mmn2, mmn3, mmn4, external_dim, timestamp_train, timestamp_test = read_cache(fname)
        print("load %s successfully" % fname)
    else:
        X_train, Y_train, X_test, Y_test, mmn1, mmn2, mmn3, mmn4, external_dim, timestamp_train, timestamp_test = TaxiNY.load_data(T=T,
                                                                                                                                   nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, preprocess_name='preprocessing.pkl',
                                                                                                                                   meta_data=True, meteorol_data=True)
        if CACHEDATA:
            cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test)

    li = []
    for v in timestamp_test[0::T]:
        v = str(v)
        li.append(v)
    print("\n days(test):", li)
    print('\n elapsed time (loading data):%.3f seconds\n' % (time.time()-ts))
    print('=' * 10)
    print('compiling model...')
    print(external_dim)
    ts = time.time()
    tf.reset_default_graph()
    #external_dim = 0
    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.lr{}.noExternal'.format(len_closeness,
                                                            len_period,
                                                            len_trend,
                                                            lr
                                                            )
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min'
    )
    print('\n exapsed time(compiling model):%.3f seconds\n' % (time.time() - ts))

    print('=' * 10)
    print('training model...')

    ts = time.time()
    # number of samples
    X_train1 = np.array(X_train[0])
    ns = X_train1.shape[0]
    X_train1 = np.reshape(X_train1, (ns, len_closeness, map_height, map_width, nb_flow))
    X_train2 = np.array(X_train[1])
    X_train2 = np.reshape(X_train2, (ns, len_period, map_height, map_width, nb_flow))
    X_train3 = np.array(X_train[2])
    X_train3 = np.reshape(X_train3, (ns, len_trend, map_height, map_width, nb_flow))
    X_train4 = np.array(X_train[3])
    X_train = [X_train1, X_train2, X_train3, X_train4]
    #X_train = [X_train1, X_train2, X_train3]
    """
    print("train1", X_train1.shape, "train2 ", X_train2.shape, \
          "train3 ", X_train3.shape, "train4 ", X_train4.shape)
    """
    X_test1 = np.array(X_test[0])
    X_test1 = np.reshape(X_test1, (len_test, len_closeness, map_height, map_width, nb_flow))
    X_test2 = np.array(X_test[1])
    X_test2 = np.reshape(X_test2, (len_test, len_period, map_height, map_width, nb_flow))
    X_test3 = np.array(X_test[2])
    X_test3 = np.reshape(X_test3, (len_test, len_trend, map_height, map_width, nb_flow))
    X_test4 = np.array(X_test[3])
    X_test = [X_test1, X_test2, X_test3, X_test4]
    #X_test = [X_test1, X_test2, X_test3]
    print('X_train1 shape: ', np.shape(X_train1), 'X_train2 shape: ', np.shape(X_train2), 'X_train3 shape: ', \
          np.shape(X_train3), 'X_train4 shape: ', np.shape(X_train4))
    print('X_test1 shape: ', np.shape(X_test1), 'X_test2 shape: ', np.shape(X_test2), 'X_test3 shape:', \
          np.shape(X_test3), 'X_test4 shape: ', np.shape(X_test4)
          )

    Y_train1 = np.array(Y_train[0])
    ns = Y_train1.shape[0]
    Y_train1 = np.reshape(Y_train1, (ns, map_height, map_width, nb_flow))
    Y_train2 = np.array(Y_train[1])
    Y_train2 = np.reshape(Y_train2, (ns, map_height, map_width, nb_flow))
    Y_train3 = np.array(Y_train[2])
    Y_train3 = np.reshape(Y_train3, (ns, map_height, map_width, nb_flow))
    Y_train4 = np.array(Y_train[3])
    Y_train4 = np.reshape(Y_train4, (ns, map_height, map_width, nb_flow))
    Y_train = [Y_train1, Y_train2, Y_train3, Y_train4]
    #Y_train = [Y_train4]

    Y_test1 = np.array(Y_test[0])
    ns = Y_test1.shape[0]
    Y_test1 = np.reshape(Y_test1, (ns, map_height, map_width, nb_flow))
    Y_test2 = np.array(Y_test[1])
    Y_test2 = np.reshape(Y_test2, (ns, map_height, map_width, nb_flow))
    Y_test3 = np.array(Y_test[2])
    Y_test3 = np.reshape(Y_test3, (ns, map_height, map_width, nb_flow))
    Y_test4 = np.array(Y_test[3])
    Y_test4 = np.reshape(Y_test4, (ns, map_height, map_width, nb_flow))
    Y_test = [Y_test1, Y_test2, Y_test3, Y_test4]
    #Y_test = [Y_test4]

    history = model.fit(
        X_train, Y_train,
        epochs=nb_epoch,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping, model_checkpoint],
        #callbacks=[model_checkpoint],
        verbose=1
    )
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)
    ), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)
    ), 'wb'))
    print('\n elapsed time(training):%.3f seconds\n' % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')
    ts = time.time()
    score = model.evaluate(X_train, Y_train,
                           batch_size=24, verbose=0)


    print("score:", len(score))
    print("score is:", score)
    """
    print('Train score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[1], score[1], score[1] * (mmn4._max - mmn4._min) / 2.))
    """
    print('Train score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
    score[5], score[5], score[5] * (mmn1._max - mmn1._min) / 2.))
    print('Train score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[6], score[6], score[6] * (mmn2._max - mmn2._min) / 2.))
    print('Train score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[7], score[7], score[7] * (mmn3._max - mmn3._min) / 2.))
    print('Train score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[8], score[8], score[8] * (mmn4._max - mmn4._min) / 2.))

    ts = time.time()
    """
    score = model.evaluate(
        X_test, Y_test, batch_size=24, verbose=0
    )
    """
    filepath = 'H:\\NewYork Taxi Data\\Prediciton CNNLSTM'
    res = model.predict(X_test, batch_size=24, verbose=0)
    print(np.shape(res))
    res_1 = res[0]
    res_1 = np.squeeze(res_1)
    res_1 = np.reshape(res_1, (168, 100))
    res_1 = (res_1+1)/2
    res_1 = res_1*(mmn1._max-mmn1._min)+mmn1._min
    #res_1 = np.sum(res_1, axis=1)
    #res_1 = np.squeeze(res_1)
    Write_To_File2(os.path.join(filepath, '5min_prediction.txt'), res_1)
    res_2 = res[1]
    res_2 = np.squeeze(res_2)
    res_2 = np.reshape(res_2, (168, 100))
    res_2 = (res_2 + 1) / 2
    res_2 = res_2 * (mmn2._max - mmn2._min) + mmn2._min
    #res_2 = np.sum(res_2, axis=1)
    #res_2 = np.squeeze(res_2)
    Write_To_File2(os.path.join(filepath, '15min_prediction.txt'), res_2)
    res_3 = res[2]
    res_3 = np.squeeze(res_3)
    res_3 = np.reshape(res_3, (168, 100))
    res_3 = (res_3 + 1) / 2
    res_3 = res_3 * (mmn3._max - mmn3._min) + mmn3._min
    #res_3 = np.sum(res_3, axis=1)
    #res_3 = np.squeeze(res_3)
    Write_To_File2(os.path.join(filepath, '30min_prediction.txt'), res_3)
    res_4 = res[3]
    res_4 = np.squeeze(res_4)
    res_4 = np.reshape(res_4, (168, 100))
    res_4 = (res_4+1)/2
    res_4 = res_4*(mmn4._max-mmn4._min)+mmn4._min
    #res_4 = np.sum(res_4, axis=1)
    #res_4 = np.squeeze(res_4)
    #np.savetxt(os.path.join(filepath, '60min_prediction.txt'), res_4, newline='\n')
    Write_To_File2(os.path.join(filepath, '60min_prediction.txt'), res_4)
    #print(np.shape(res_4))
    #print(np.shape(Y_test))
    #print(Y_test1)
    Y_test1 = np.squeeze(Y_test1)
    Y_test1 = np.reshape(Y_test1, (168, 100))
    Y_test1 = (Y_test1+1)/2
    Y_test1 = Y_test1*(mmn1._max-mmn1._min)+mmn1._min
    #Y_test1 = np.sum(Y_test1, axis=1)
    #Y_test1 = np.squeeze(Y_test1)
    Write_To_File2(os.path.join(filepath, '5min_ground truth.txt'), Y_test1)
    Y_test2 = np.squeeze(Y_test2)
    Y_test2 = np.reshape(Y_test2, (168, 100))
    Y_test2 = (Y_test2 + 1) / 2
    Y_test2 = Y_test2 * (mmn2._max - mmn2._min) + mmn2._min
    #Y_test2 = np.sum(Y_test2, axis=1)
    #Y_test2 = np.squeeze(Y_test2)
    Write_To_File2(os.path.join(filepath, '15min_ground truth.txt'), Y_test2)
    Y_test3 = np.squeeze(Y_test3)
    Y_test3 = np.reshape(Y_test3, (168, 100))
    Y_test3 = (Y_test3 + 1) / 2
    Y_test3 = Y_test3 * (mmn3._max - mmn3._min) + mmn3._min
    #Y_test3 = np.sum(Y_test3, axis=1)
    #Y_test3 = np.squeeze(Y_test3)
    Write_To_File2(os.path.join(filepath, '30min_ground truth.txt'), Y_test3)
    Y_test4 = np.squeeze(Y_test4)
    Y_test4 = np.reshape(Y_test4, (168, 100))
    Y_test4 = (Y_test4 + 1) / 2
    Y_test4 = Y_test4 * (mmn4._max - mmn4._min) + mmn4._min
    #Y_test4 = np.sum(Y_test4, axis=1)
    #Y_test4 = np.squeeze(Y_test4)
    Write_To_File2(os.path.join(filepath, '60min_ground truth.txt'), Y_test4)

    """
    print('Test score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[1], score[1], score[1] * (mmn4._max - mmn4._min) / 2.))
    """
    print('Test score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[5], score[5], score[5] * (mmn1._max - mmn1._min) / 2.))
    print('Test score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[6], score[6], score[6] * (mmn2._max - mmn2._min) / 2.))
    print('Test score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[7], score[7], score[7] * (mmn3._max - mmn3._min) / 2.))
    print('Test score:%.6f rmse (norm):%.6f rmse (real):%.6f' % (
        score[8], score[8], score[8] * (mmn4._max - mmn4._min) / 2.))
    print('\n elapsed time (eval):%.3f seconds\n' % (time.time() - ts))



if __name__ == '__main__':
        main()


