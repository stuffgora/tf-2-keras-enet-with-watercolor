# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:39:17 2019

@author: abraverm
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from src.enet_wc_keras_model import autoencoder_wc, transfer_weights
from src.camvid_dataset_load import prepare_for_training, create_dataset 
from src.camvid_dataset_load import create_coco_dataset, create_coco_test_set

from src.camvid_dataset_load import  median_frequency_balancing
from src.enetcfg import EnetCfg

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
#tf.debugging.set_log_device_placement(True)

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


class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    
    def __init__(self,filepath, monitor='val_loss', verbose=1, 
                 save_best_only=True, save_weights_only=True, 
                 mode='auto', ):
        
        super(EpochModelCheckpoint, self).__init__(filepath=filepath,monitor=monitor,
             verbose=verbose,save_best_only=save_best_only,
             save_weights_only=save_weights_only, mode=mode)
        
        self.ckpt = tf.train.Checkpoint(completed_epochs=tf.Variable(0,trainable=False,dtype='int32'))
        ckpt_dir = f'{os.path.dirname(filepath)}/tf_ckpts'
        self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=3)
        
    def on_epoch_begin(self,epoch,logs=None):        
        self.ckpt.completed_epochs.assign(epoch)
        self.manager.save()
        print( f"Epoch checkpoint {self.ckpt.completed_epochs.numpy()}  saved to: {self.manager.latest_checkpoint}" ) 
        print(logs)


def callbacks(log_dir, checkpoint_dir, model_name):
    tb = TensorBoard(log_dir=log_dir,
                     histogram_freq=1,
                     write_graph=True,
                     write_images=True)
    best_model = os.path.join(checkpoint_dir, f'{model_name}_best.hdf5')
    save_best = EpochModelCheckpoint( best_model  )
    
    checkpoint_file = os.path.join(checkpoint_dir, 'weights.' + model_name + '.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoints = ModelCheckpoint(
        filepath=checkpoint_file,   
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto'
        #period=1
        )

    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    return [tb, save_best ]
    #return [tb, save_best, checkpoints]
    #return []


class WaitedCategoricalCrossentropy (tf.keras.losses.CategoricalCrossentropy):
    def __init__(self,class_w):
        super(WaitedCategoricalCrossentropy, self).__init__(from_logits=True,name='waighted_categorical_crossentropy') 
        self.c_w = class_w
    def __call__(self, y_true, y_pred, sample_weight=None):
        weights = y_true * self.c_w
        weights = tf.reduce_sum(weights, -1)        
        return super(WaitedCategoricalCrossentropy, self).__call__(y_true, y_pred, sample_weight)*weights


def get_class_normalization_dic(dataset=cfg.dataset_name):
    cd_d = None     
    if dataset ==  'camvid': 
#        class_weights = median_frequency_balancing()
#        cw_d = {}        
#        for i,c in enumerate(class_weights):
#             cw_d[i] = c
        cw_d = {0: 0.0159531051456976, 1: 0.011580246710544183, 2: 0.22857586995014328, 3: 0.009042348126826805, 4: 0.05747495410789924, 5: 0.025342723815993118, 6: 0.16389458162792303, 7: 0.2807956777529651, 8: 0.0931421249518621186, 9: 0.9930486077110527676, 10: 0.85542331331773912, 11: 0.0001}
    if dataset ==  'coco': 
        cw_d = {0: 0.0, 1: 0.05, 2: 0.1, 3: 0.05, 4: 0.05, 5: 0.1, 6: 0.3, 7: 0.1, 8: 0.1, 9: 0.6, 10: 0.05, 11: 0.05, 12: 0.2  }
        cw_d = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0  }
        #cw_d = None
    return cd_d


def get_checkpoint_log_dir():
    experiment_dir = os.path.join('models', cfg.dataset_name, cfg.model_name)
    log_dir = os.path.join(experiment_dir, 'logs')
    checkpoint_dir = os.path.join(experiment_dir, 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir, log_dir


def model_preload(model):
    # ------------ Model preload ------------
    checkpoint_dir, _ = get_checkpoint_log_dir()
    ckpt = tf.train.Checkpoint(completed_epochs=tf.Variable(0,trainable=False,dtype='int32'))
    manager = tf.train.CheckpointManager(ckpt, f'{checkpoint_dir}/tf_ckpts', max_to_keep=3)    
    best_weights = os.path.join(checkpoint_dir, f'{cfg.model_name}_best.hdf5') 
    #best_weights = os.path.join(checkpoint_dir, f'enet_best.hdf5') #ckpt
    #best_weights = 'wc1_preloaded_coco.hdf5'
    best_weights = 'evet_no_wc_preload_from_coco.hdf5'
    print(f'Tryigg to load model {best_weights}')
    if os.path.exists(best_weights):
        print(f'Loading model {best_weights}')
        model.load_weights(best_weights)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print(f"Restored epoch ckpt from {manager.latest_checkpoint}, value is ",ckpt.completed_epochs.numpy())
        else:
            print("Initializing from scratch.")
    else:        
        model = transfer_weights(model)
    print('Done loading {} model!'.format(cfg.model_name))
#    print("TODO Not Trainable !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#    for idx in range(10,len(model.layers)):
#        model.layers[idx].trainable = False
    return model, ckpt


def get_metrics():
    new_metrics = [ #tf.keras.metrics.MeanIoU(num_classes=nc),
                    tf.keras.metrics.Precision(), ]
    metrics=['accuracy', 'mean_squared_error']
    metrics += new_metrics
    return metrics
    

def get_optimizer():
    initial_learning_rate = 5e-4 #0.1
    #decay_steps = int(num_epochs_before_decay * num_steps_per_epoch) ~100*100
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10*2*(50*10),#    (steps_in_s*batch)
        decay_rate=1e-1, #0.96,
        staircase=True)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
    optimizer='adadelta'
    return optimizer
    

def get_loss():
    #loss = WaitedCategoricalCrossentropy(class_w=class_weights)
    loss = "categorical_crossentropy"
    return loss


def get_model():
    dw = cfg.image_width
    dh = cfg.image_height
    
    optimizer = get_optimizer()
    loss = get_loss()
    metrics = get_metrics()
    
    model, model_name = autoencoder_wc(nc=cfg.num_classes, input_shape=(dw, dh),
                                         loss=loss,
                                         optimizer=optimizer, 
                                         metrics=metrics, 
                                         wc_in_encoder=cfg.wc_in_encoder,
                                         wc_in_decoder=cfg.wc_in_decoder)
    return model

        
def get_train_val_data(dataset = cfg.dataset_name):
    dw = cfg.image_width
    dh = cfg.image_height
    nc = cfg.num_classes #12 #get_classes_nmber()
    class_weights = get_class_normalization_dic()
    if dataset ==  'coco':  
        val_ds = create_coco_dataset(dataDir='../../../cocodataset', dataType='val2017', im_w=dw, im_h=dh)
        # 118300 semples
        train_ds = create_coco_dataset(dataDir='../../../cocodataset', dataType='train2017', im_w=dw, im_h=dh)
    elif dataset == 'camvid':
        data_dir = "../dataset/train"                
        train_ds = create_dataset(data_dir,im_w=dw,im_h=dh, num_classes=nc,reshape=None,class_w=class_weights)
        data_dir = "../dataset/val"
        val_ds = create_dataset(data_dir,im_w=dw,im_h=dh, num_classes=nc,reshape=None,class_w=class_weights) 
    
    if cfg.concat_ds > 0:
        train_ds.concatenate(val_ds)

    train_ds = prepare_for_training(train_ds, batch_size=cfg.batch_size, cache=None, shuffle_buffer_size=2)
    print("infinit dataset !!!!!!!!!!!!!!!!!!!!!")
    train_ds=train_ds.repeat()
    val_ds   = prepare_for_training(val_ds,   batch_size=cfg.val_batch_size, cache=None, shuffle_buffer_size=2)
    return train_ds, val_ds

        
def train( ):
    print(f'Preparing to train on {cfg.dataset_name} data...')
    
    autoenc = get_model()
    autoenc, ckpt = model_preload(autoenc) 
    train_ds, val_ds = get_train_val_data(dataset = cfg.dataset_name)
    
    # Class Waight balancing        
    cw_d = get_class_normalization_dic(dataset=cfg.dataset_name)
    checkpoint_dir, log_dir = get_checkpoint_log_dir()
    # checkpoint for epoch counter
    if cfg.initial_epoch is None:
        completed_epochs = ckpt.completed_epochs.numpy()
    else :
        completed_epochs = cfg.initial_epoch  
    # ---------- Fit Model - Training-----------------
    history = autoenc.fit(
        x=train_ds,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps,
        class_weight=cw_d,
        verbose=1,
        callbacks=callbacks(log_dir, checkpoint_dir, cfg.model_name ),        
        validation_data=val_ds,
        validation_steps=cfg.val_steps,
        initial_epoch=completed_epochs
    )  
    print('\nhistory dict:', history.history)
    return autoenc


def evaluate():
    model = get_model()
    model, _ = model_preload(model) 
    
    #-------- TODO get_dataset -------
    #test_ds = create_coco_test_set(im_w=cfg.image_width, im_h=cfg.image_height)
    val_ds = create_coco_dataset(dataType='val2017', im_w=cfg.image_width, im_h=cfg.image_height) 
    test_ds = val_ds
    test_ds = model.batch(100)
    #--------------------------------
    results = model.evaluate(test_ds, steps=100) 
    print('\nTEST Evaluete:\ntest loss, test acc, test mse, test precision:', results)
    return model


label_to_colours =    {0: [128,128,128],
                     1: [128,0,0],
                     2: [192,192,128],
                     3: [128,64,128],
                     4: [60,40,222],
                     5: [128,128,0],
                     6: [192,128,128],
                     7: [64,64,128],
                     8: [64,0,128],
                     9: [64,128,0], #Pedestrian [64,64,0]
                     10: [0,128,192],   # Bicyclist = [0,128,192]              
                     11: [0,0,0],
                     12: [120,56,70]}

def onehot_2_rgb(lbl):
    shape = lbl.shape
    img = np.zeros([shape[0], shape[1],3], dtype=np.uint16)
    for i in range(shape[0]):
        for j in range(shape[1]):
            img[i][j] =label_to_colours[ lbl[i][j] ]
    return img


def visualize_prediction(pred,lbl,img):
    pred  = onehot_2_rgb( tf.math.argmax(pred,axis=-1).numpy() )
    lable = onehot_2_rgb( tf.math.argmax(lbl,axis=-1).numpy() )
    lable = np.reshape(lable, [256,256,3])
    fig, ((ax1,ax2,ax3)) = plt.subplots(1,3)
    for ax in [ax1,ax2,ax3]: ax.axis('off')
    #plt.figure(figsize=(10,10))
    ax1.imshow(pred) #,cmap='winter')
    ax2.imshow(lable)
    ax3.imshow(img)
    plt.show()
    

def predict():
    model = get_model()
    model, _ = model_preload(model)  
    #test_ds = create_coco_test_set(im_w=cfg.image_width, im_h=cfg.image_height)
    #val_ds = create_coco_dataset(dataType='val2017', im_w=cfg.image_width, im_h=cfg.image_height) 
    _, val_ds = get_train_val_data()
    test_ds = val_ds
    #test_ds = test_ds.batch(10)
    for ims ,lbls in test_ds.take(1):
        predictions = model.predict(ims)
        for pred,lbl,img in zip(predictions,lbls,ims):
            visualize_prediction(pred,lbl,img)
    return model



if __name__ == '__main__':
#    cfg.train_flow = 0
#    cfg.predict_flow  = 0
#    
    if cfg.train_flow > 0:
        print(f'Training ENet model: {cfg.model_name}, with datatset: {cfg.dataset_name}')
        trained_model = train()
    if cfg.predict_flow > 0:
        prediction_model = predict()
    
    if 0>0:
        cfg.image_width = 256
        cfg.image_height = 256
        cfg.wc_in_encoder = 0
        cfg.wc_in_decoder = 1
        model = get_model()
        print("model.summary()")
        model.summary()
        print(' ---------- len(model.trainable_variables) ------ ')
        print('trainable_variables #:',len(model.trainable_variables))
        print('layers #:',len(model.layers))
        tf.keras.utils.plot_model(model, f'{cfg.model_name}_model_with_shape_info.png', show_shapes=True)
    #model preload
    if 0>0:
        cfg.image_width = 256
        cfg.image_height = 256
        cfg.wc_in_encoder = 0
        cfg.wc_in_decoder = 0
        # first 7 layers : inp -> conv2d->maxpull->concat->[0:3] init -> conv2d->norm->prelu 
        enet_model = get_model()
        best_weights = 'models/coco/enet_no_wc_256x256/weights/enet_no_wc_256x256_best.hdf5'
        enet_model.load_weights(best_weights)
        
        cfg.wc_in_encoder = 1
        wc_model = get_model()
        offset = len(wc_model.layers) - len(enet_model.layers)
        for idx in range(7,len(enet_model.layers)):
           wc_model.layers[idx+offset].set_weights(enet_model.layers[idx].get_weights())
           wc_model.layers[idx+offset].trainable = False
        
    
    
    
    
    
    
    
    
    
    
