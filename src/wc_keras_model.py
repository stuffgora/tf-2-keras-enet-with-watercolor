# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:41:43 2019

@author: abraverm
"""

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math as math
import numpy as np

from src.enetcfg import EnetCfg
flags = EnetCfg()
#flags.DEFINE_string('wc_activatin', None, "None - use no activation, <actv> - use <actv> in water collor midle layers, for example 'tanh/relu'" )
flags.default_enet_cfg()
cfg = flags.parse_args()



def get_laplace_mtrx(nx,ny,d=1):
    dx = d # defusion speed
    dy = d
    diag_block = np.eye(ny)*(-2/dx**2+-2/dy**2)
    diag_block = diag_block + np.diag(np.ones((ny-1))/dy**2,1)
    diag_block = diag_block + np.diag(np.ones((ny-1))/dy**2,-1)
    Matrix = np.kron(np.eye(nx),diag_block)
    Matrix = Matrix + np.diag(np.ones([(nx-1)*(ny)])/dx**2, ny )
    Matrix = Matrix + np.diag(np.ones([(nx-1)*(ny)])/dx**2, -(ny ))
#    M_inv = np.linalg.inv(Matrix)
    return Matrix


def get_oponent_filter() :
    s2 = math.sqrt(2.0)
    s6 = math.sqrt(6.0)
    a = 0.2989
    b = 0.5780
    c = 0.1140
    flt = [[1/s2, -1/s2,  0   ],
           [1/s6,  1/s6, -2/s6],
           [  a ,    b ,  c   ]
          ]
    return np.transpose(flt)


def opponent_init(shape, dtype='float32', partition_info=None):
    return tf.convert_to_tensor ( get_oponent_filter(), dtype =tf.float32, name ='oponent_kerenl' )


def inv_opponent_init(shape, dtype=None,partition_info=None):
    kernel = np.linalg.inv(get_oponent_filter())
    return tf.convert_to_tensor ( kernel , dtype =tf.float32, name ='oponent_kerenl' )


def gabor_odd_init(shape, dtype=None,partition_info=None):
    kernel = np.zeros([1,2,3,3])
    kernel[:,:,0,0] = [[-1.0,1.0]]
    kernel[:,:,1,1] = [[-1.0,1.0]]
    kernel[:,:,2,2] = [[-1.0,1.0]]
    return tf.convert_to_tensor ( kernel, dtype='float32' )


def gabor_even_init(shape, dtype=None,partition_info=None):
    kernel = np.zeros([1,3,3,3])
    kernel[:,:,0,0] = [[-1.0,2.0,-1.0]]
    kernel[:,:,1,1] = [[-1.0,2.0,-1.0]]
    kernel[:,:,2,2] = [[-1.0,2.0,-1.0]]
    return tf.convert_to_tensor ( kernel, dtype='float32')


def wc_waits_layer(inputs, depth):
    factor = 2**depth
    shape = inputs.shape.as_list()
    
    prmd_l = layers.AvgPool2D(pool_size=factor)(inputs)
    prmd_l = layers.Conv2D(3, (1,3), padding='same', activation='tanh',
                  kernel_initializer=gabor_even_init)(prmd_l)
    prmd_l = layers.UpSampling2D( size=factor, interpolation='nearest')(prmd_l)
    if shape != prmd_l.shape.as_list() :
        pad_h = shape[1] - prmd_l.shape.as_list()[1]
        h_pad_h = int(pad_h/2)
        pad_w = shape[2] - prmd_l.shape.as_list()[2]
        h_pad_w = int(pad_w/2)
        padding = ((h_pad_h, pad_h - h_pad_h) , (h_pad_w, pad_w - h_pad_w) )            
        prmd_l = layers.ZeroPadding2D(padding = padding )(prmd_l)
    prmd_l = tf.abs(prmd_l)
    w = tf.abs(tf.stack([inputs,prmd_l], axis=4))
    w = tf.reduce_max(w,axis=4)
    return w


def get_wc_waits(inputs, prmd_levels):
    w = inputs
    for idx in range(1,prmd_levels):
        w = wc_waits_layer(w, idx)
    return w
        
        
def dct2(x):  
    ry = tf.signal.dct(x,type=1)
    r = tf.signal.dct( tf.transpose(ry), type=1)
    return tf.transpose(r) 


def idct2(x):
    return dct2(x)  


def get_lam_mat(m,n,h=0.7):
    pi=np.pi
    lam_n = np.arange(1,n+1)
    lam_n = -4/h**2 * (np.sin((lam_n*pi) / (2*(n - 1))))**2 
    lam_m = np.arange(1,m+1)
    lam_m = -4/h**2 * (np.sin((lam_m*pi) / (2*(m -1))))**2 #$compute $\lambda_x$
    lam_mat_np = np.ones([m,n])
    for i in np.arange(0,m):
        for j in np.arange(0,n):
            lam_mat_np[i,j] = (lam_mat_np[i,j]) / (lam_m[i] + lam_n[j])
    #return tf.convert_to_tensor(lam_mat_np, dtype=tf.float32)
    return lam_mat_np
 
      
def fft_poisson_2d(f, lam_mat ):
    m,n = lam_mat.shape
    f_bar = idct2(f)            # f_bar= fourier transform of f
    normlz = (2.0/n + 1.0)*(2.0/m + 1.0)
    f_bar = f_bar * normlz  #Normalize
    u_bar = layers.Multiply()([f_bar,lam_mat])
    u = dct2(u_bar)                #sine transform back
    normlz = 2.0/((n-1.0)*(m-1.0))
    u = u * normlz         #normalize 
    return u


def poisson_3d (f, lam_mat ):
    f_t = tf.transpose(f,[2,0,1])
    fixed_poisson_2d = lambda x : fft_poisson_2d(x, lam_mat)
    res_t = tf.map_fn(fixed_poisson_2d,f_t,parallel_iterations=True)
    return tf.transpose(res_t,[1,2,0])


class PoisonSolver3D(tf.keras.layers.Layer):

    def __init__(self,in_shape, h=0.7): # **kwargs):
        #kwargs['autocast']=False
        super(PoisonSolver3D, self).__init__() #**kwargs)
        _,w,h,c = in_shape
        self.lamda_mat = tf.constant(get_lam_mat(w,h))
        
    def get_config(self):
        super(PoisonSolver3D, self).get_config()

    def call(self, inp):
        #_,w,h,c = inp.shape
        #self.lamda_mat=tf.constant(get_lam_mat(w,h))
        fixed_poisson = tf.function(lambda x : poisson_3d(x, self.lamda_mat) )
        return tf.map_fn(fixed_poisson,inp)

class PoisonSolverLplMat(tf.keras.layers.Layer):

    def __init__(self,in_shape,d=1.0): # **kwargs):
        super(PoisonSolverLplMat, self).__init__() #**kwargs)
        _,w,h,c = in_shape
        lpl_mat = tf.constant(get_laplace_mtrx(w,h,d=d), dtype='float32')
        print("lpl shape is" , lpl_mat.shape)
        self.inv_lpl_mat = tf.constant(tf.linalg.inv(lpl_mat))
        print("lpl inv shape is" ,self.inv_lpl_mat.shape)
        
    def get_config(self):
        super(PoisonSolverLplMat, self).get_config()
    
    def laplace_inv(self,x):
        w,h,c = x.shape
        r = tf.reshape(x, [-1, c])
        print("r shape is" ,r.shape)
        print("lpl shape is" ,self.inv_lpl_mat.shape)
        r = tf.matmul(self.inv_lpl_mat, r)
        return  tf.reshape(r, x.shape)
        
        
    def call(self, inp):
        return tf.map_fn(self.laplace_inv,inp)


def get_wc_model (in_shape, 
                  pad=3,
                  alpha = 0.6,
                  beta = 0.3,
                  pyramid_depth = 3,
                  defusion_speed = 0.5,
                  reuse=None,
                  is_training=True,
                  scope='wc_model',
                  lpl_mat = None,
                  activation ='tanh',
                  ):
#    if lam_mat is None:
#        #_,m,n,c = inputs.shape #
#        m,n,c = in_shape
#        tf_get_lam_mat = tf.function(get_lam_mat)
#        lam_mat = tf_get_lam_mat( m + 2*pad, n + 2*pad , h=defusion_speed)
#    
#    if inputs is None:
#        inputs =  layers.Input(shape=in_shape)
    inputs =  layers.Input(shape=in_shape)
    #with tf.variable_scope(scope, reuse=reuse):
    alpha = tf.Variable(alpha)
    beta  = tf.Variable(beta)

    #in_opponent = layers.Dense(3, input_shape=(3,), use_bias=False, activation='tanh',
    in_opponent = layers.Dense(3,  use_bias=False, activation=activation,
                     kernel_initializer=opponent_init, dtype='float32') (inputs)
    #show_tensor_img(in_opponent[0],"opponent")

    f = layers.ZeroPadding2D(padding = pad )(in_opponent)
    im_dx = layers.Conv2D(3,(1,2), padding='same', activation=activation,
                          kernel_initializer=gabor_odd_init,dtype='float32')(f)
    im_dy = tf.transpose(f,[0,2,1,3])
    #Permute((2, 1), input_shape=(10, 64))
    im_dy = layers.Conv2D(3,(1,2), padding='same', activation=activation,
                          kernel_initializer=gabor_odd_init)(im_dy)
    
    wx = get_wc_waits(in_opponent, pyramid_depth)
    wy = get_wc_waits(tf.transpose(in_opponent, [0,2,1,3]), pyramid_depth )
    wx = layers.ZeroPadding2D(padding = pad )(wx)
    wy = layers.ZeroPadding2D(padding = pad )(wy)
    #show_tensor_img(wx[0],"wx")
    #show_tensor_img(wy[0],"wy")
    im_dx_w = tf.math.multiply(im_dx,wx)
    im_dy_w = tf.math.multiply(im_dy,wy)
                               
    trig_xx = alpha*im_dx + beta*im_dx_w
    trig_yy = alpha*im_dy + beta*im_dy_w
    
    trig_xx = layers.Conv2D(3,(1,2), padding='same', activation=activation,
                          kernel_initializer=gabor_odd_init,dtype='float32')(trig_xx)                          
    trig_yy = layers.Conv2D(3,(1,2), padding='same', activation=activation,
                          kernel_initializer=gabor_odd_init,dtype='float32')(trig_yy)
    
    trig_yy = tf.transpose(trig_yy,[0,2,1,3])       
    div_trig = trig_xx + trig_yy
    #show_tensor_img(div_trig[0],"div_trig")
    
    #lam_mat = get_lam_mat( m + 2*pad, n + 2*pad , h=defusion_speed)
    #fixed_poisson = tf.function(lambda x : poisson_3d(x, lam_mat) )
    #res = tf.map_fn(fixed_poisson,div_trig)
    #res = div_trig
    #TODO fix
    #res = dct_try(div_trig)
    if lpl_mat is None:
        res=PoisonSolver3D(in_shape=div_trig.shape,h=defusion_speed)(div_trig)
    else:
        res = PoisonSolverLplMat(in_shape=div_trig.shape,d=1)(div_trig)
    #res = layers.Lambda(dct_try)(div_trig)
    #res = layers.Conv2D(8,(2,2), padding='same', activation='tanh',)(div_trig)
    #res = layers.Conv2D(3,(2,2), padding='same', activation='tanh',)(res)
    #res = layers.Conv2D(3,(2,2), padding='same', activation='tanh',)(res)
    #show_tensor_img(res[0],"result img")
    res= layers.Cropping2D(cropping=pad)(res)
    
    model = tf.keras.Model(inputs=inputs, outputs=res)
    return model
    
def show_tensor_img(tf_img, title = 'no title'):
    #init_op = tf.initialize_all_variables()
    #with tf.Session() as sess:
        #sess.run(init_op)
        #im = sess.run(tf_img)
    my_show_image(tf_img,title)


def my_show_image(img,title):
    
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title(title)
    plt.show()

def decode_img(img,im_w=64,im_h=64):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  img = tf.image.resize(img, [im_w, im_w])
  return  img

if __name__ == '__main__':
    
    #with tf.device('/CPU:0'):

        im_w = 64
        im_h = 64
        im_c = 3
        pad = 4
       
        img_file = '../gray_square.png'
        img = tf.io.read_file(img_file)
        list_ds = tf.data.Dataset.list_files(img_file) #str(data_dir/'*/*')
        for f in list_ds.take(1):
            print(f.numpy())

        #my_img_rgb = tf.image.decode_image('gray_square.png', dtype=tf.float32)
        my_img_rgb = decode_img(img)
        my_show_image(my_img_rgb, "source img")
        im1 = tf.image.resize_with_pad(my_img_rgb, im_h, im_w)
        im1 = tf.reshape(im1,[1,im_h,im_w,im_c], name="input_reshaped")
        show_tensor_img(im1[0],"Source image")
        
        model = get_wc_model( in_shape=[im_w,im_h,im_c], pad=pad, lpl_mat=cfg.wc_lpl_mat, activation=cfg.wc_activation  )
        
        res = model([im1,im1,im1])
        res_rgb = layers.Dense(3, input_shape=(3,), use_bias=False, kernel_initializer=inv_opponent_init,dtype='float32')(res)
        #res_rgb = tf.keras.activations.sigmoid(res_rgb)
        #res_rgb = tf.keras.activations.tanh(res_rgb)
        res_rgb = tf.keras.activations.relu(res_rgb)
    
        res = tf.reshape(res,[res.shape[1],res.shape[2],im_c])
        show_tensor_img(res,"result image")
        
        res_rgb = tf.reshape(res_rgb,[res_rgb.shape[1],res_rgb.shape[2],im_c])
        show_tensor_img(res_rgb,"result RGB image")
        
        loss='categorical_crossentropy',
        optimizer='adadelta'
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'mean_squared_error'])

        
    
