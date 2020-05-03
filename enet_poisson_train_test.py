#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 01:34:41 2020

@author: ddd
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from src.enet_wc_keras_model import autoencoder_wc, enet_poisson, wc_poisson, wc_poisson_small
from src.enetcfg import EnetCfg
from src.camvid_dataset_load import create_poisson_rand_dataset

from PIL import Image
from src.watercolor02_np_fft import to_water_collor, my_img_show



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

flags = EnetCfg()
flags.default_enet_cfg(flags=flags)
cfg = flags.parse_args()


def rand_img(m,n):
#    img = '../stars_clean.png'    
#    x = np.array(Image.open(img).convert("RGB"))/256.0 
    #for i in range(100000):
    while True:
        x = np.random.rand(m,n,3)
        y = to_water_collor(x)
        yield (x,y)

def get_rand_data(m,n):
#    return tf.data.Dataset.from_generator(
#            rand_img, 
#            output_types=(tf.float32, tf.float32),
#            output_shapes=(tf.TensorShape([m,n,3]), tf.TensorShape([m,n,3])),
#            args=(m,n)
#            )
    #data_dir = 'dataset/rand_wc'
    data_dir = 'dataset/rand_abstr2'
    abstr2_ds = create_poisson_rand_dataset(data_dir, im_w=m, im_h=n )
    data_dir = 'dataset/rand_wc'
    rand_ds = create_poisson_rand_dataset(data_dir, im_w=m, im_h=n )
    data_dir = 'dataset/rand_abstr'
    val_ds = create_poisson_rand_dataset(data_dir, im_w=m, im_h=n )
    
    train_ds = abstr2_ds.concatenate(rand_ds).shuffle(3000, reshuffle_each_iteration=False)
    
    return train_ds, val_ds

def visualize_poisson_ds(data_dir = 'dataset/rand_abstr2'):
    m,n = (256,256)
    ds = create_poisson_rand_dataset(data_dir, im_w=m, im_h=n )
    #plt.figure(figsize=(10,10))
    axs = plt.subplots(5,2)
    for row in axs:
        im ,lbl  = ds.take(1)
        row[0].imshow(im[0,:,:,:])
        row[1].imshow(lbl[0,:,:,:])

    

def get_optimizer():
    initial_learning_rate = 5e-3 #0.1
    #decay_steps = int(num_epochs_before_decay * num_steps_per_epoch) ~100*100
#    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#        initial_learning_rate,
#        decay_steps=100, # 100 #500, #10*2*(50*10),#    (steps_in_s*batch)
#        decay_rate=0.96, #1e-1, #0.96,
#        staircase=True)
    
#    pl_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
#            initial_learning_rate = 0.1 , decay_steps=1000, 
#            end_learning_rate=0.000001, power=1.0,
#            cycle=True, name=None
#    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, epsilon=1e-8)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.3, nesterov=True)
    #optimizer = tf.optimizers.Adadelta(learning_rate=5e-5, rho=1e-1) #, epsiif data_transform is not None :
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

    #optimizer='adadelta'
    return optimizer
   
def get_model(cfg=cfg):
    dw = cfg.image_width
    dh = cfg.image_height
    
    #optimizer = 'adam' #'Adagrad' #'SGD' # get_optimizer() #'SGD' # 'adam' #
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, epsilon=1e-8)
    optimizer = get_optimizer()
    loss = 'logcosh' #'cosine_similarity' #'kullback_leibler_divergence' #'logcosh' stars_clean
    #'mean_absolute_error' #'mean_squared_error' #get_loss()
    metrics=['accuracy', 'mean_squared_error'] #get_metrics()
    def eval_model(img = 'gray_square.png' ):
    model_dir = 'saved_model/poisson_model'
    checkpoint_dir = f'{model_dir}/training_checkpoints'
    model = tf.keras.models.load_model(model_dir)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    m, n = (256,256)
       
    x = np.array(Image.open(img).resize((n, m)).convert("RGB"))/256.0
    
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    y = np.squeeze(y, axis=0)
    x = np.squeeze(x, axis=0)
    z = to_water_collor(x)
#    my_img_show(y[:,:,0], "chn0")
#    my_img_show(y[:,:,1], "chn1")
#    my_img_show(y[:,:,2], "chn2")
#    my_img_show(x, "src")
#    my_img_show(y, "res")
#    
    imgs = [(y[:,:,0], "Result chn0"),
    (y[:,:,1], "Result chn1"),
    (y[:,:,2], "Result chn2"),
    (x, "RGB source image"),
    (y, "RGB result"),
    (z, "WC Graound truth")]
    fig, axs = plt.subplots(nrows=2, ncols=3, subplot_kw={'xticks': [], 'yticks': []})
    for ax, im in  zip(axs.flat, imgs):
        ax.imshow(im[0])
        ax.set_title(im[1])
    #model, model_name = autoencoder_wc(
    model, model_name = enet_poisson(
    #model, model_name = wc_poisson_small( 
    #model, model_name = wc_poisson(
            nc=3, 
            input_shape=(dw, dh),
            loss=loss,
            optimizer=optimizer, 
            metrics=metrics )                                      
    return model


def train( ):
    cfg.dataset_name = 'rand_abstract'
    print(f'Preparing to train on {cfg.dataset_name} data...')
    
    model_dir = 'saved_model/poisson_model'
    # Define the checkpoint directory to store the checkpoint
    checkpoint_dir = f'{model_dir}/training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
   
    autoenc = get_model()
    
    #model.save('saved_model/poisson_model')
    
    #autoenc.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    #pre_trained_model = tf.keras.models.load_model(model_dir)
    #autoenc.set_weights(pre_trained_model.get_weights())
    
    # Callback for printing the LR at the end of each epoch.
    class PrintLR(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        print(f'\nLearning rate for epoch {epoch + 1} is {autoenc.optimizer.lr.numpy()}')

    # Function for decaying the learning rate.
    # You can define any decay function you need.
    def decay(epoch):
        ref = epoch%10
        if   ref < 3:from watercolor02_np_fft import to_water_collor
            return 1e-3
        elif ref >= 3 and ref < 7:
            return 1e-4
        elif ref >= 7 and ref < 9:
            return 1e-5
        else:
            return 1e-6
    
      
    def eval_model(img = 'gray_square.png' ):
    model_dir = 'saved_model/poisson_model'
    checkpoint_dir = f'{model_dir}/training_checkpoints'
    model = tf.keras.models.load_model(model_dir)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    m, n = (256,256)
       
    x = np.array(Image.open(img).resize((n, m)).convert("RGB"))/256.0
    
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    y = np.squeeze(y, axis=0)
    x = np.squeeze(x, axis=0)
    z = to_water_collor(x)
#    my_img_show(y[:,:,0], "chn0")
#    my_img_show(y[:,:,1], "chn1")
#    my_img_show(y[:,:,2], "chn2")
#    my_img_show(x, "src")
#    my_img_show(y, "res")
#    
    imgs = [(y[:,:,0], "Result chn0"),
    (y[:,:,1], "Result chn1"),
    (y[:,:,2], "Result chn2"),
    (x, "RGB source image"),
    (y, "RGB result"),
    (z, "WC Graound truth")]
    fig, axs = plt.subplots(nrows=2, ncols=3, subplot_kw={'xticks': [], 'yticks': []})
    for ax, im in  zip(axs.flat, imgs):
        ax.imshow(im[0])
        ax.set_title(im[1])
    m = cfg.image_width
    n = cfg.image_height
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=f'{model_dir}/logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                           save_weights_only=True,
                                           save_best_only=True,
                                           monitor='val_loss'),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]
    
    train_ds, val_ds = get_rand_data(m,n)
    #train_ds = train_ds.repeat(-1)
    #val_ds   = get_rand_data(m,n)
    
    # ---------- Fit Model - Training-----------------
    completed_epochs = 40
    history = autoenc.fit(
        x=train_ds.batch(20),
        epochs=300, # 10, #200, #cfg.epochs,
        #steps_per_epoch=200, #cfg.steps,
        #class_weight=cw_d,
        verbose=1,
        callbacks=callbacks, #callbacks(log_dir, checkpoint_dir, cfg.model_name ),        
        validation_data=val_ds.batch(20), #val_ds.batch(2),
        #validation_steps=4 #cfg.val_steps,
        initial_epoch=completed_epochs
    )  
    print('\nhistory dict:', history.history)
    return autoenc, history


def eval_model(img = 'gray_square.png' ):
    model_dir = 'saved_model/poisson_model'
    checkpoint_dir = f'{model_dir}/training_checkpoints'
    model = tf.keras.models.load_model(model_dir)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    m, n = (256,256)
       
    x = np.array(Image.open(img).resize((n, m)).convert("RGB"))/256.0
    
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    y = np.squeeze(y, axis=0)
    x = np.squeeze(x, axis=0)
    z = to_water_collor(x, alpha = 0.9, beta = 0.2, h = 1.0)
#    my_img_show(y[:,:,0], "chn0")
#    my_img_show(y[:,:,1], "chn1")
#    my_img_show(y[:,:,2], "chn2")
#    my_img_show(x, "src")
#    my_img_show(y, "res")
#    
    imgs = [(y[:,:,0], "Result chn0"),
    (y[:,:,1], "Result chn1"),
    (y[:,:,2], "Result chn2"),
    (x, "RGB source image"),
    (y, "RGB result"),
    (z, "WC Graound truth")]
    fig, axs = plt.subplots(nrows=2, ncols=3, subplot_kw={'xticks': [], 'yticks': []})
    for ax, im in  zip(axs.flat, imgs):
        ax.imshow(im[0])
        ax.set_title(im[1])
        
if __name__ == '__main__':
    cfg.image_width  = 256
    cfg.image_height = 256
    m = cfg.image_width
    n = cfg.image_height
    

    model_dir = 'saved_model/poisson_model'
    
    img = 'stars_clean.png'    
    x = np.array(Image.open(img).resize((n, m)).convert("RGB"))/256.0
    
    #model = get_model()
    #model.save('saved_model/poisson_model')
    
    model, history = train( )
    model.save(model_dir)
    
    #new_model = tf.keras.models.load_model('saved_model/my_model') 
     
    #model = tf.keras.models.load_model(model_dir)
    
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    y = np.squeeze(y, axis=0)
    my_img_show(y[:,:,0], "res0")
    my_img_show(y[:,:,1], "res1")
    my_img_show(y[:,:,2], "res2")
    my_img_show(x[0,:,:,:], "src")
    my_img_show(y, "res")
    print('\nhistory dict:', history.history['val_accuracy'])
 
    # 0.529
    
    
    
    
    
    
    
    
    
    
    
    