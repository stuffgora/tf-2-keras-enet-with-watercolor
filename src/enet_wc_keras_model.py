# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:56:35 2019

@author: abraverm
"""
import tensorflow as tf

from tensorflow.keras.layers import Concatenate,Add
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import ZeroPadding2D, Conv2DTranspose # Convolution2D, 
from tensorflow.keras.layers import Permute, SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Softmax

from src.wc_keras_model import get_wc_model 
from src.enetcfg import EnetCfg

flags = EnetCfg()
flags.default_enet_cfg()
cfg = flags.parse_args()

WcLayer = lambda in_shape : get_wc_model(in_shape=in_shape, pyramid_depth=3,lpl_mat=cfg.wc_lpl_mat, activation=cfg.wc_activation )


weight_decay=2e-4
conv2_r = tf.keras.regularizers.l2(weight_decay)


#Convolution2D =  layers.Convolution2D(kernel_regularizer=tf.keras.regularizers.l1(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay))
class Convolution2D(tf.keras.layers.Conv2D):
    def __init__(self,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=conv2_r,
    bias_regularizer=conv2_r,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
    ):
        super(Convolution2D, self).__init__(
                filters,
                kernel_size,
                strides,
                padding,
                data_format,
                dilation_rate,
                activation,
                use_bias,
                kernel_initializer,
                bias_initializer,
                kernel_regularizer,
                bias_regularizer,
                activity_regularizer,
                kernel_constraint,
                bias_constraint,
                **kwargs
                ) 
        
        


def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, conv_stride=(2, 2), wc=None,wc_dropout_rate = 0.2):
    merged = []
    #---- WC 1 added ----------
    if wc == 1 : # 3 conv channels with wc
        wc_net = Convolution2D(3, [nb_row, nb_col], padding='same', strides=conv_stride)(inp)
        _,w,h,c = wc_net.shape
        wc_net = WcLayer(in_shape=[w,h,c])(wc_net)
        
        wc_net = SpatialDropout2D(wc_dropout_rate)(wc_net)
        
        nb_filter -= 3
        merged += [wc_net]
      
    #-----------
    conv = Convolution2D(nb_filter, [nb_row, nb_col], padding='same', strides=conv_stride)(inp)
    max_pool = MaxPooling2D()(inp)
    
    #---- WC 2 added ----------
    if wc == 2 : # 3 maxpull channels with wc
        wc_net = max_pool
        _,w,h,c = wc_net.shape
        wc_net = WcLayer(in_shape=[w,h,c])(wc_net)
        wc_net = tf.keras.activations.tanh(wc_net)
        wc_net = SpatialDropout2D(wc_dropout_rate)(wc_net)
        max_pool =  wc_net
    #-----------
    merged += [conv, max_pool]
    merged = Concatenate(axis=3)(merged)
    
     #---- WC 3 added ----------
    if wc == 3 :
        wc_net = Convolution2D(3, [nb_row, nb_col], padding='same', strides=conv_stride)(inp)
        _,w,h,c = wc_net.shape
        wc_net = WcLayer(in_shape=[w,h,c])(wc_net)
        wc_net = Convolution2D(nb_filter+3, [1, 1], padding='same')(wc_net)
        wc_net = SpatialDropout2D(wc_dropout_rate)(wc_net)
        #wc_net = BatchNormalization(momentum=0.1)(wc_net)
        wc_net = tf.keras.activations.tanh(wc_net)
        merged = Add()([merged, wc_net])
    #-----------
    
    
    return merged


def bottleneck(inp, output, internal_scale=4, use_relu=True, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = int(output / internal_scale)
    encoder = inp

    ## 1x1
    input_stride = 2 if downsample else 1  # the first 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Convolution2D(internal, [input_stride, input_stride], padding='same', strides=[input_stride, input_stride], use_bias=False)(encoder)
    ## Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder) # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    ## conv
    if not asymmetric and not dilated:
        encoder = Convolution2D(internal, [3, 3], padding='same')(encoder)
    elif asymmetric:
        encoder = Convolution2D(internal, [1, asymmetric], padding='same', use_bias=False)(encoder)
        encoder = Convolution2D(internal, [asymmetric, 1], padding='same')(encoder)
    elif dilated:
        encoder = Convolution2D(internal, [3, 3], dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise(Exception('You shouldn\'t be here'))

    ## Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder) # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    
    ## 1x1
    encoder = Convolution2D(output, [1, 1], padding='same', use_bias=False)(encoder)
    
    ## Batch normalization + Spatial dropout
    encoder = BatchNormalization(momentum=0.1)(encoder) # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)
        other = Permute((1, 3, 2))(other)
        pad_featmaps = output - inp.get_shape().as_list()[3]
        other = ZeroPadding2D(padding=((0, 0), (0, pad_featmaps)))(other)
        other = Permute((1, 3, 2))(other)

    encoder = Add()([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    return encoder


def build_encoder(inp, dropout_rate=0.01, wc=None):
    enet = initial_block(inp,wc=wc)
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for i in range(4):
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate) # bottleneck 1.i
    
    enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for i in range(2):
        enet = bottleneck(enet, 128)  # bottleneck 2.1
        enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = bottleneck(enet, 128)  # bottleneck 2.5
        enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8
    return enet

def bottleneck_decoder(inp, output, upsample=False, reverse_module=False):
    internal = int(output / 4)
    input_stride = 2 if upsample else 1
    
    x = Convolution2D(internal, (input_stride, input_stride), padding='same', use_bias=False)(inp)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU()(x)
    if not upsample:
        x = Convolution2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        b, w, h, nb_filters = inp.get_shape().as_list()
        x = Conv2DTranspose(internal, (3, 3),  padding='same', strides=(2, 2) )(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU()(x)

    x = Convolution2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = inp
    if inp.get_shape()[-1] != output or upsample:
        other = Convolution2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module:
            other = UpSampling2D(size=(2, 2))(other)
        
    if not upsample or reverse_module:
        x = BatchNormalization(momentum=0.1)(x)
    else:
        return x
    
    decoder = Add()([x, other])
    decoder = PReLU()(decoder)
    return decoder

def decoder_wc_layer(inp, wc=1):
    
    _,w,h,out_c = inp.shape
    wc_net = Convolution2D(3, (1, 1), padding='same', use_bias=False)(inp)
    wc_net = BatchNormalization(momentum=0.1)(wc_net)
    _,w,h,c = wc_net.shape
    wc_net = WcLayer(in_shape=[w,h,c])(wc_net)
    # TODO add net with padding 
    wc_net = Convolution2D(out_c, (1, 1), padding='same', use_bias=False)(wc_net)
    wc_net = BatchNormalization(momentum=0.1)(wc_net)
    wc_net = Add()([wc_net, inp])
    wc_net = PReLU()(wc_net)   
    return wc_net
    
    
def build_decoder(encoder, nc, in_shape, dropout_rate=0.1, wc=None):
    # print(encoder.get_shape().as_list())
    enet = bottleneck_decoder(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = bottleneck_decoder(enet, 64)  # bottleneck 4.1
    enet = bottleneck_decoder(enet, 64)  # bottleneck 4.2
    # ------ added WC ----------
    if wc is not None:
        enet = decoder_wc_layer(enet)
    # --------------------------
    enet = bottleneck_decoder(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = bottleneck_decoder(enet, 16)  # bottleneck 5.1
    enet = Conv2DTranspose(nc, [2, 2],  padding='same', strides=(2, 2))(enet)
    return enet


def transfer_weights(model, weights=None):
    '''
    Always trains from scratch; never transfers weights
    '''
    print('ENet has found no compatible pretrained weights! Skipping weight transfer...')
    return model



def autoencoder_wc(nc, input_shape,
                loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy', 'mean_squared_error'],
                wc_in_encoder=None,
                wc_in_decoder=None):
    inp =  Input(shape=(input_shape[0], input_shape[1], 3))
    enet = build_encoder(inp,wc=wc_in_encoder)
    enet = build_decoder(enet, nc=nc, in_shape=input_shape,wc=wc_in_decoder)
    
    #enet = Reshape((data_shape, nc))(enet)
    enet = Softmax()(enet)
    model = tf.keras.Model(inputs=inp, outputs=enet)
    
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    name = 'enet'

    return model, name


if __name__ == '__main__':
    with tf.device('/CPU:0'):
        autoencoder_wc, name = autoencoder_wc(nc=2, input_shape=(512, 512))
        tf.keras.utils.plot_model(autoencoder_wc, 'enet_wc_net.png', show_shapes=True)
    print("Done!")
    
  
    
    
    
    
    
    
    
    
    
    
    
    