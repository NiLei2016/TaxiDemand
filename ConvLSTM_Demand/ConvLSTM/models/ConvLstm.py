from __future__ import print_function
from keras.layers.core import Dense
from keras.layers.core import Reshape, Activation, Flatten
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import ConvLSTM2D, Convolution2D
from keras.layers.recurrent import GRU, LSTM, SimpleRNN
from keras.layers import Input, add
from keras.models import Model
from keras.regularizers import l1, l2,l1_l2
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K
nb_flow = 1
class iLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def compute_output_shape(self, input_shape):
        return input_shape
class MyLayer(Layer):
    def __int__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.random.normal((input_dim,))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]
    def call(self, inputs, **kwargs):
        return inputs * self.W
    def get_output_shape(self, input_shape):
        return input_shape

def seqGRU_CPTM(c_conf=(3,1,10,10), p_conf=(3,1,10,10),t_conf=(3,1,10,10),external_dim=23):
    main_inputs = []
    outputs = []
    main_outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, channel, map_height, map_width = conf
            input = Input(shape=(len_seq, map_height, map_width, channel))
            reshape_1 = Reshape(target_shape=(len_seq, -1))(input)
            main_inputs.append(input)
            lstm_1 = GRU(units=channel * map_height * map_width, activation='relu', return_sequences=True)(reshape_1)
            batch_1 = BatchNormalization()(lstm_1)
            drop_1 = Dropout(0.2)(batch_1)
            lstm_2 = GRU(units=channel * map_height * map_width, activation='relu',return_sequences=True)(drop_1)
            activation_2 = Activation('relu')(lstm_2)
            batch_2 = BatchNormalization()(activation_2)
            lstm_3 = GRU(units=channel * map_height * map_width, activation='relu',return_sequences=True)(batch_2)
            batch_3 = BatchNormalization()(lstm_3)
            lstm_4 = GRU(units=channel * map_height * map_width, activation='relu')(batch_3)
            batch_4 = BatchNormalization()(lstm_4)
            outputs.append(batch_4)
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:

            new_outputs.append(iLayer()(output))
        main_output = add(new_outputs)
    main_output = Reshape((10, 10, 1))(main_output)
    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=channel * map_width * map_height)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((map_height, map_width, channel))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim: ', external_dim)
    output1 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output1 = Activation('tanh')(output1)
    output2 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output2 = Activation('tanh')(output2)
    output3 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output3 = Activation('tanh')(output3)
    output4 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output4 = Activation('tanh')(output4)
    main_outputs.append(output1)
    main_outputs.append(output2)
    main_outputs.append(output3)
    main_outputs.append(output4)
    model = Model(inputs=main_inputs, outputs=main_outputs)
    return model
def seqRNN_CPTM(c_conf=(3,1,10,10), p_conf=(3,1,10,10),t_conf=(3,1,10,10),external_dim=23):
    main_inputs = []
    outputs = []
    main_outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, channel, map_height, map_width = conf
            input = Input(shape=(len_seq, map_height, map_width, channel))
            reshape_1 = Reshape(target_shape=(len_seq, -1))(input)
            main_inputs.append(input)
            rnn_1 = SimpleRNN(units=channel * map_height * map_width, return_sequences=True)(reshape_1)
            activation_1 = Activation('relu')(rnn_1)
            batch_1 = BatchNormalization()(activation_1)
            rnn_2 = SimpleRNN(units=channel * map_height * map_width, return_sequences=True)(batch_1)
            activation_2 = Activation('relu')(rnn_2)
            batch_2 = BatchNormalization()(activation_2)
            rnn_3 = SimpleRNN(units=channel * map_height * map_width, return_sequences=True)(batch_2)
            activation_3 = Activation('relu')(rnn_3)
            batch_3 = BatchNormalization()(activation_3)
            rnn_4 = SimpleRNN(units=channel * map_height * map_width)(batch_3)
            activation_4 = Activation('relu')(rnn_4)
            batch_4 = BatchNormalization()(activation_4)
            outputs.append(batch_4)
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for ouput in outputs:
            new_outputs.append(iLayer()(ouput))
        main_output = add(new_outputs)
    main_output = Reshape((10, 10, 1))(main_output)
    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=channel * map_width * map_height)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((map_height, map_width, channel))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim: ', external_dim)
    output1 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output1 = Activation('tanh')(output1)
    output2 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output2 = Activation('tanh')(output2)
    output3 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output3 = Activation('tanh')(output3)
    output4 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output4 = Activation('tanh')(output4)
    main_outputs.append(output1)
    main_outputs.append(output2)
    main_outputs.append(output3)
    main_outputs.append(output4)
    model = Model(inputs=main_inputs, outputs=main_outputs)
    return model
def seqLSTM_CPTM(c_conf=(3,1,10,10), p_conf=(3,1,10,10),t_conf=(3,1,10,10),external_dim=23):
    main_inputs = []
    outputs = []
    main_outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, channel, map_height, map_width = conf
            input = Input(shape=(len_seq, map_height, map_width, channel))
            reshape_1 = Reshape(target_shape=(len_seq, -1))(input)
            main_inputs.append(input)
            lstm_1 = LSTM(units=channel * map_height * map_width, activation='relu', return_sequences=True)(reshape_1)
            batch_1 = BatchNormalization()(lstm_1)
            drop_1 = Dropout(0.2)(batch_1)
            lstm_2 = LSTM(units=channel * map_height * map_width, activation='relu', return_sequences=True)(drop_1)
            batch_2 = BatchNormalization()(lstm_2)
            drop2 = Dropout(0.2)(batch_2)
            lstm_3 = LSTM(units=channel * map_height * map_width, activation='relu', return_sequences=True)(drop2)
            batch_3 = BatchNormalization()(lstm_3)
            drop3 = Dropout(0.2)(batch_3)
            lstm_4 = LSTM(units=channel * map_height * map_width, activation='relu')(drop3)
            batch_4 = BatchNormalization()(lstm_4)
            outputs.append(batch_4)
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for ouput in outputs:
            new_outputs.append(iLayer()(ouput))
        main_output = add(new_outputs)
    main_output = Reshape((10, 10, 1))(main_output)
        # fusing with external component
    if external_dim != None and external_dim > 0:
            # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=channel * map_width * map_height)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((map_height, map_width, channel))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim: ', external_dim)
    output1 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output1 = Activation('tanh')(output1)
    output2 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output2 = Activation('tanh')(output2)
    output3 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output3 = Activation('tanh')(output3)
    output4 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output4 = Activation('tanh')(output4)
    main_outputs.append(output1)
    main_outputs.append(output2)
    main_outputs.append(output3)
    main_outputs.append(output4)
    model = Model(inputs=main_inputs, outputs=main_outputs)
    return model
def seqCNN_LSTM_CPTM(c_conf=(3,1,10,10), p_conf=(3,1,10,10),t_conf=(3,1,10,10),external_dim=23):
    main_inputs = []
    outputs = []
    main_outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, channel, map_height, map_width = conf
            input = Input(shape=(len_seq, map_height, map_width, channel))
            main_inputs.append(input)
            reshape_1 = Reshape(target_shape=(map_height, map_width, -1))(input)
            conv_1 = Convolution2D(filters=64, kernel_size=(3, 3), padding='same')(reshape_1)
            activation_1 = Activation('relu')(conv_1)
            batch_1 = BatchNormalization()(activation_1)
            drop_1 = Dropout(0.2)(batch_1)
            conv_2 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(drop_1)
            activation_2 = Activation('relu')(conv_2)
            batch_2 = BatchNormalization()(activation_2)
            drop_2 = Dropout(0.2)(batch_2)
            reshape_3 = Reshape(target_shape=(-1, map_height*map_width*channel))(drop_2)
            lstm_3 = LSTM(units=channel * map_height * map_width, activation='relu', return_sequences=True)(reshape_3)
            batch_3 = BatchNormalization()(lstm_3)
            drop_3 = Dropout(0.2)(batch_3)
            lstm_4 = LSTM(units=channel * map_height * map_width, activation='relu')(drop_3)
            batch_4 = BatchNormalization()(lstm_4)
            activation_4 = Activation('relu')(batch_4)
            outputs.append(activation_4)
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for ouput in outputs:
            new_outputs.append(iLayer()(ouput))
        main_output = add(new_outputs)
    main_output = Reshape((10, 10, 1))(main_output)
        # main_output = Reshape((10, 10, 1))(main_output)
        # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=channel * map_width * map_height)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((map_height, map_width, channel))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim: ', external_dim)
    output1 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output1 = Activation('tanh')(output1)
    output2 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output2 = Activation('tanh')(output2)
    output3 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output3 = Activation('tanh')(output3)
    output4 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output4 = Activation('tanh')(output4)
    main_outputs.append(output1)
    main_outputs.append(output2)
    main_outputs.append(output3)
    main_outputs.append(output4)
    model = Model(inputs=main_inputs, outputs=main_outputs)
    return model
def seqConv_CPTM(c_conf=(3,1,10,10), p_conf=(3,1,10,10),t_conf=(3,1,10,10),external_dim=23):
    main_inputs = []
    outputs = []
    main_outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, channel, map_height, map_width = conf
            input = Input(shape=(len_seq, map_height, map_width, channel))
            main_inputs.append(input)
            reshape_1 = Reshape(target_shape=(map_height, map_width, -1))(input)
            conv_1 = Convolution2D(filters=64, kernel_size=(3, 3), padding='same')(reshape_1)
            activation_1 = Activation('relu')(conv_1)
            batch_1 = BatchNormalization()(activation_1)
            drop_1 = Dropout(0.2)(batch_1)
            conv_2 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(drop_1)
            activation_2 = Activation('relu')(conv_2)
            batch_2 = BatchNormalization()(activation_2)
            drop_2 = Dropout(0.2)(batch_2)
            conv_3 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(drop_2)
            activation_3 = Activation('relu')(conv_3)
            batch_3 = BatchNormalization()(activation_3)
            drop_3 = Dropout(0.2)(batch_3)
            conv_4 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(drop_3)
            activation_4 = Activation('relu')(conv_4)
            outputs.append(activation_4)
    if len(outputs) == 1:
            main_output = outputs[0]
    else:
        new_outputs = []
        for ouput in outputs:
            new_outputs.append(iLayer()(ouput))
        main_output = add(new_outputs)
    #main_output = Reshape((10, 10, 1))(main_output)
        # fusing with external component
    if external_dim != None and external_dim > 0:
            # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=channel * map_width * map_height)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((map_height, map_width, channel))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim: ', external_dim)
    output1 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output1 = Activation('tanh')(output1)
    output2 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output2 = Activation('tanh')(output2)
    output3 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output3 = Activation('tanh')(output3)
    output4 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output4 = Activation('tanh')(output4)
    main_outputs.append(output1)
    main_outputs.append(output2)
    main_outputs.append(output3)
    main_outputs.append(output4)
    model = Model(inputs=main_inputs, outputs=main_outputs)
    return model
#
def seqConvLSTM_CPTM( c_conf=(12, 1, 10, 10), p_conf=(3, 1, 10, 10),t_conf=(3, 1, 10, 10), external_dim=23):
    main_inputs = []
    outputs = []
    main_outputs = []
    for conf in [c_conf, p_conf,t_conf]:
        if conf is not None:
            len_seq, channel, map_height, map_width = conf
            input = Input(shape=(len_seq, map_height, map_width, channel))
            main_inputs.append(input)
            # ConvLSTM1
            convLstm1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                                   )(input)
            activation1 = Activation('relu')(convLstm1)
            batch1 = BatchNormalization()(activation1)
            dropout2 = Dropout(0.2)(batch1)
            convLstm2 = ConvLSTM2D(filters=channel, kernel_size=(3, 3), padding='same')(dropout2)
            activation2 = Activation('relu')(convLstm2)
            """"
            dropout3 = Dropout(0.2)(convLstm2)
            convLstm3 = ConvLSTM2D(filters=channel, kernel_size=(3, 3), padding='same', return_sequences=True)(dropout3)
            dropout4 = Dropout(0.2)(convLstm3)
            convLstm4 = ConvLSTM2D(filters=channel, kernel_size=(3, 3), padding='same')(dropout4)
            """
            outputs.append(activation2)
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = add(new_outputs)
    #main_output = Reshape((10, 10, 1))(main_output)
    #fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=channel*map_width*map_height)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((map_height, map_width, channel))(activation)
        print(external_output)
        main_output = add([main_output, external_output])
        print(main_output)
    else:
        print('external_dim: ', external_dim)
    print(main_output)
    output1 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output1 = Activation('tanh')(output1)
    output2 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output2 = Activation('tanh')(output2)
    output3 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output3 = Activation('tanh')(output3)
    output4 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output4 = Activation('tanh')(output4)
    main_outputs.append(output1)
    main_outputs.append(output2)
    main_outputs.append(output3)
    main_outputs.append(output4)
    model = Model(inputs=main_inputs, outputs=main_outputs)
    return model
if __name__ == '__main__':
    with K.get_session():
        model = seqCNN_LSTM_CPTM(external_dim=23)
        model.summary()
    K.clear_session()







