from __future__ import print_function
from keras.layers.core import  Dense
from keras.layers import Input, Activation,  Reshape, add
from keras.layers import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import  backend as K
#from keras.utils.visualize_util import plot


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
class iLayer(Layer):
    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        initial_wright_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_wright_value)
        self.trainable_weights = [self.W]
    def call(self, x, mask=None):
        return x * self.W
    def compute_output_shape(self, input_shape):
        return input_shape

def _shortcut(input,residual):
    return add([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, padding='same')(activation)
    return f
def _residual_unit(nb_filter,init_subsample=(1,1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f
def ResUnits(residual_unit,nb_filter,repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = _residual_unit(nb_filter=nb_filter, init_subsample=init_subsample)(input)
        return input
    return f
def stresnet(c_conf=(3,1,10,10), p_conf=(3, 1, 10, 10), t_conf=(3, 1, 10, 10), external_dim=23, nb_residual_unit=12):
    ''''
    C-Temporal Closeness
    P-Period
    T-Trend
    conf=(len_seq,nb_flow,map_height,map_width)
    external_dim
    '''
    #main input
    main_inputs = []
    main_outputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, channel, map_height, map_width = conf
            input = Input(shape=(len_seq, map_height, map_width, channel))
            main_inputs.append(input)
            reshape_1 = Reshape(target_shape=(map_height, map_width, -1))(input)
            conv_1 = Convolution2D(filters=64, kernel_size=(3, 3), padding='same')(reshape_1)
            #[nb_residual_unit] Residual Units
            residual_output = ResUnits(nb_residual_unit, nb_filter=64,
                                     repetations=nb_residual_unit)(conv_1)
            #Conv2

            activation = Activation('relu')(residual_output)
            #activation = Activation('relu')(conv1)
            conv2 = Convolution2D(
                filters=channel, kernel_size=(3, 3), padding='same'
            )(activation)
            outputs.append(conv2)
    #parameter-matrix-base fusion
    if(len(outputs)) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = add(new_outputs)

        #fusing with external component
    if external_dim != None and external_dim > 0:
            #external input
            external_input = Input(shape=(external_dim,))
            main_inputs.append(external_input)
            embedding = Dense(units=10)(external_input)
            embedding = Activation('relu')(embedding)
            h1 = Dense(units=channel*map_height*map_width)(embedding)
            activation = Activation('relu')(h1)
            external_output = Reshape((map_height, map_width, channel))(activation)
            main_output = add([main_output, external_output])
    else:
            print('external_dim:', external_dim)
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

if __name__=='__main__':
    with K.get_session():
        model = stresnet(external_dim=28, nb_residual_unit=12)
        model.summary()
    K.clear_session()

