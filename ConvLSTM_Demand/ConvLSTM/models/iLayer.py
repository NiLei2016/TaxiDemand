from __future__ import print_function
from keras.layers.core import Dense
from keras.layers.core import Reshape, Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import ConvLSTM2D, Convolution2D
from keras.layers import Input, add
from keras.models import Model
from keras.layers import Add
from keras.engine.topology import Layer
import numpy as np
from keras import backend as K

class iLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        initial_wright_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_wright_value)
        self.trainable_weights = [self.W]
    def call(self, x, mask = None):
        return x*self.W
    def get_output_shape_for(self, input_shape):
        return input_shape
def seqConvLSTM_CPTM(c_conf = (3,1,10,10),p_conf = (3,1,10,10),t_conf = (3,1,10,10),external_dim = 23):
    main_inputs = []
    outputs = []
    main_outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, channel, map_height, map_width = conf
            input = Input(shape=(len_seq, map_height, map_width, channel))
            main_inputs.append(input)
            # ConvLSTM1
            convLstm1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(input)
            activation1 = Activation('relu')(convLstm1)
            batch1 = BatchNormalization()(activation1)
            dropout1 = Dropout(0.2)(batch1)
            convLstm2 = ConvLSTM2D(filters=channel, kernel_size=(3, 3), padding='same')(dropout1)
            outputs.append(convLstm2)
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for ouput in outputs:
            new_outputs.append(iLayer()(ouput))
        main_output = add(new_outputs)
    #fusing with external component
    if external_dim != None and external_dim>0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=channel*map_width*map_height)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((map_height, map_width, channel))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim: ', external_dim)
    output1 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output1 = Activation('tanh')(output1)
    output2 = Convolution2D(filters=channel, kernel_size=(3,3),padding='same')(main_output)
    output2 = Activation('tanh')(output2)
    output3 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output3 = Activation('tanh')(output3)
    output4 = Convolution2D(filters=channel, kernel_size=(3, 3), padding='same')(main_output)
    output4 = Activation('tanh')(output4)
    main_outputs.append(output1)
    main_outputs.append(output2)
    main_outputs.append(output3)
    main_outputs.append(output4)
    model = Model(inputs=main_inputs, outputs= main_outputs)
    return model
if __name__ == '__main__':
    with K.get_session():
        model = seqConvLSTM_CPTM(external_dim=23)
        model.summary()
    K.clear_session()







