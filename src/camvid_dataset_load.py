#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:23:18 2019

@author: ddd
"""

import tensorflow as tf
#tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

from pycocotools.coco import COCO

#import IPython.display as display
#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import pathlib
import os

def show_batch(image_batch, label_batch=""):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      #plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
  plt.show()


def process_path(file_path):
  #label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  #return img, label
  return img


def show_img_lbl(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  ax = plt.subplot(1,2,1)
  plt.imshow(image_batch)
  plt.axis('off')
  plt.subplot(1,2,2)
  plt.imshow(label_batch[:,:,0])
  #plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
  plt.axis('off')
  plt.show()
    


def prepare_for_training(ds, batch_size=1, cache=None, shuffle_buffer_size=100):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  #ds = ds.repeat()
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  
  ds = ds.batch(batch_size)
  #ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # Repeat forever
  
  #ds = ds.batch(batch_size)
  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


def median_frequency_balancing(image_files='', num_classes=12):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c

    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.

    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    - num_classes(int): the number of classes of pixels in all images

    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    #Initialize all the labels key with a list value
    label_to_frequency_dict = {}
    for i in range(num_classes):
        label_to_frequency_dict[i] = []

    for n in range(len(image_files)):
        image = imageio.imread(image_files[n])

        #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
        for i in range(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                label_to_frequency_dict[i].append(class_frequency)

    class_weights = []

    #Get the total pixels to calculate total_frequency later
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)

    for i, j in label_to_frequency_dict.items():
        j = sorted(j) #To obtain the median, we got to sort the frequencies

        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)

    #Set the last class_weight to 0.0 as it's the background class
    class_weights[-1] = 0.0

    return class_weights

def process_img_from_path(file_path,im_w=None,im_h=None,chn=3,dtype=tf.float32, ns=None, reshape=1):
  #label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=chn)
  if im_w is not None:
    #img = tf.image.resize_with_pad(img, [im_w, im_h], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.resize(img, [im_w, im_h], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  img = tf.image.convert_image_dtype(img, dtype)
  if ns is not None:
    shape = img.shape
    img = tf.one_hot(indices=img, depth=ns)
    
    img = tf.reshape(img, (shape[0], shape[1], ns))
    if reshape:
        img = tf.reshape(img, (-1, ns) )
  return img


def y2w(onehot_lbls,w=None):
    if w is None:
        return 1.
    weights = onehot_lbls * w
    weights = tf.reduce_sum(weights, -1)
    return weights
    
def flip_data(x,y):
    #x,y = d
    if tf.random.uniform([],0,2,dtype="int32") > 0:
        #x = tf.image.flip_left_right(x)
        #y = tf.image.flip_left_right(y)
        x = tf.reverse(x,[0])
        y = tf.reverse(y,[0])
    if tf.random.uniform([],0,2,dtype="int32") > 0:
        x = tf.reverse(x,[1])
        y = tf.reverse(y,[1])
    return (x,y)


def get_labeled_dataset(data_dir, file, im_w=None,im_h=None, num_classes=12,  reshape=1, class_w=None):
    lbl_data_dir = data_dir+"annot"     
    data_dir     = pathlib.Path(data_dir) 
    lbl_data_dir = pathlib.Path(lbl_data_dir) 
    
    x = f"{data_dir}/{file}"
    y = f"{lbl_data_dir}/{file}"
    x = process_img_from_path(x, im_w=im_w,im_h=im_h,chn=3,dtype=tf.float32)
    y = process_img_from_path(y, im_w=im_w,im_h=im_h,chn=1,dtype=tf.uint8, ns=num_classes,reshape=reshape)
    
    return x,y
    
def create_dataset_new(data_dir,im_w=None,im_h=None, num_classes=12,  reshape=1, class_w=None, data_transform=1):
  lbl_data_dir = data_dir+"annot"
  lbl_data_dir = pathlib.Path(lbl_data_dir)
  lbl_file_names = [f.name for f in pathlib.Path(lbl_data_dir).glob("*.png")]
  
  labeled_ds = tf.data.Dataset.list_files( lbl_file_names) 
    
  process_lbl_ds_path = lambda f : get_labeled_dataset(data_dir=data_dir, file=f, im_w=im_w,im_h=im_h, num_classes=num_classes,  reshape=reshape, class_w=class_w)
  labeled_ds = labeled_ds.map(process_lbl_ds_path, num_parallel_calls=AUTOTUNE)
  
  return labeled_ds


def get_coco_sup_cat_map():
    return { 'background': 0,
             'electronic': 1,
             'appliance': 2,
             'sports': 3,
             'kitchen': 4,
             'indoor': 5,
             'animal': 6,
             'food': 7,
             'furniture': 8,
             'person': 9,
             'accessory': 10,
             'outdoor': 11,
             'vehicle': 12}

def get_coco_sup_cat_id(coco,id) :
    map = get_coco_sup_cat_map()
    return map[coco.loadCats([id])[0]['supercategory']]

def get_coco_classes_nmber():   
    cat_map = get_coco_sup_cat_map()
    return len(cat_map.keys())

def validate_coco_imgs(imgIds,coco,data_dir):
    for imId in imgIds:
        im_file = coco.loadImgs([imId])[0]['file_name']
        im_file = f'{data_dir}/{im_file}'
        # wrong lbl_file = f'{data_dir}_annotImg/{im_file}.png'
        if not os.path.isfile(im_file):
           imgIds.remove(imId)
           print(f"removed imId {imId} from {data_dir} ")
    return imgIds

def id_to_img_lbl(  img_id=324158, coco=None, data_dir=None):
    nc = get_coco_classes_nmber()
    img = coco.loadImgs(img_id)[0]
    target_shape = (img['height'], img['width'],  nc)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask_one_hot = np.zeros(target_shape, dtype=np.uint8)
    mask_one_hot[:, :, 0] = 1  # every pixel begins as background
    for ann in anns:
        mask_partial = coco.annToMask(ann)
        assert mask_one_hot.shape[:2] == mask_partial.shape[:2]  # width and height match
        chan = get_coco_sup_cat_id(coco, ann['category_id']) 
        mask_one_hot[mask_partial > 0, chan] = 1
        mask_one_hot[mask_partial > 0, 0] = 0
    return  f"{data_dir}/{img['file_name']}", np.array(mask_one_hot)

def process_coco_imgs_old(file_path, lbl, im_w=None, im_h=None, dtype=tf.float32, chn=3):
  #label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=chn)
  if im_w is not None:
    #img = tf.image.resize_with_pad(img, [im_w, im_h], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.resize(img, [im_w, im_h], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lbl = tf.image.resize(lbl, [im_w, im_h], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  img = tf.image.convert_image_dtype(img, dtype)
  #lbl = tf.one_hot(indices=img, depth=ns)
  #img = tf.reshape(img, (shape[0], shape[1], ns))  
  return img, lbl

def process_coco_imgs(data, im_w=None, im_h=None, dtype=tf.float32, chn=3):
    x_file = data[0]
    y_file = data[1]
    nc = get_coco_classes_nmber()
    img = tf.io.read_file(x_file)
    #img = tf.image.decode_jpeg(img, channels=chn)
    img = tf.image.decode_image(img, channels=chn,dtype=dtype) #, expand_animations=False)
    lbl = tf.io.read_file(y_file)
    lbl = tf.image.decode_image(lbl, channels=1, dtype='uint8', expand_animations=False)
    if im_w is not None:
        img = tf.image.resize_with_pad(img, im_h, im_w, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        lbl = tf.image.resize_with_pad(lbl, im_h, im_w, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #img = tf.image.convert_image_dtype(img, dtype)
    shape = lbl.shape
    #lbl = tf.image.convert_image_dtype(lbl, dtype='uint8')
    lbl = tf.one_hot(indices=lbl, depth=nc, dtype='uint8')
    lbl = tf.reshape(lbl, (shape[0], shape[1], nc))  
    return img, lbl

#def create_coco_dataset(dataDir='../../cocodataset', dataType='val2017', subdir ='annotations_trainval2017', im_w=256, im_h=256):
def create_coco_dataset(dataDir='../../cocodataset', dataType='val2017', im_w=256, im_h=256):
    subdir ='annotations_trainval2017'
    annFile=f'{dataDir}/{subdir}/annotations/instances_{dataType}.json'
    data_dir = f"{dataDir}/{dataType}"
    coco=COCO(annFile)
    imgIds = coco.getImgIds(imgIds=[],catIds=[])
    imgIds = validate_coco_imgs(imgIds,coco, data_dir=data_dir)
    imgs = coco.loadImgs(imgIds)
    files = [f"{img['file_name']}" for img in imgs]
    #y_files = [ f"{data_dir}_annotImg/{img['file_name']}.png" for img in imgs]
    data = [(f"{data_dir}/{f}",f"{data_dir}_annotImg/{f}.png")  for f in files]
    np.random.shuffle(data)
#    x_ds = tf.data.Dataset.list_files( x_files, seed=0 )
#    y_ds = tf.data.Dataset.list_files( y_files, seed=0 )
#    labeled_ds = tf.data.Dataset.zip( (x_ds,y_ds) )   
    labeled_ds = tf.data.Dataset.from_tensor_slices(data)
    labeled_ds =labeled_ds.shuffle(buffer_size=10000)
    proc_imgs = lambda data : process_coco_imgs( data, im_w=im_w, im_h=im_h)
    labeled_ds = labeled_ds.map(proc_imgs)
    return labeled_ds

def create_coco_test_set(dataDir='../../cocodataset', dataType='test2017', im_w=256, im_h=256):
    subdir ='image_info_test2017'
    annFile=f'{dataDir}/{subdir}/annotations/{subdir}.json'
    data_dir = f"{dataDir}/{dataType}"
    coco=COCO(annFile)
    imgIds = coco.getImgIds(imgIds=[],catIds=[])
    imgIds = validate_coco_imgs(imgIds,coco, data_dir=data_dir)
    imgs = coco.loadImgs(imgIds)
    files = [f"{img['file_name']}" for img in imgs]
    #y_files = [ f"{data_dir}_annotImg/{img['file_name']}.png" for img in imgs]
    data = [(f"{data_dir}/{f}",f"{data_dir}_annotImg/{f}.png")  for f in files]
#    x_ds = tf.data.Dataset.list_files( x_files, seed=0 )
#    y_ds = tf.data.Dataset.list_files( y_files, seed=0 )
#    labeled_ds = tf.data.Dataset.zip( (x_ds,y_ds) )   
    labeled_ds = tf.data.Dataset.from_tensor_slices(data)
    proc_imgs = lambda data : process_coco_imgs( data, im_w=im_w, im_h=im_h)
    labeled_ds = labeled_ds.map(proc_imgs)
    return labeled_ds


    
def create_coco_dataset_old(dataDir='../../cocodataset', dataType='val2017', im_w=256, im_h=256):
    subdir ='annotations_trainval2017'
    annFile=f'{dataDir}/{subdir}/annotations/instances_{dataType}.json'
    data_dir = f"{dataDir}/{dataType}"
    coco=COCO(annFile)
    imgIds = coco.getImgIds(imgIds=[],catIds=[])
    imgIds = validate_coco_imgs(imgIds,coco, data_dir=data_dir)
    lbled_ds = tf.data.Dataset.from_tensor_slices(imgIds)
    get_labeld_images = lambda img_id :id_to_img_lbl( img_id=img_id, coco=coco, data_dir=data_dir)
    lbled_ds = lbled_ds.map(get_labeld_images) 
    proc_imgs = lambda x,y : process_coco_imgs(x,y, im_w=im_w, im_h=im_h)
    lbled_ds = lbled_ds.map(proc_imgs)
    return lbled_ds    
    
def create_dataset(data_dir,im_w=None,im_h=None, num_classes=12,  reshape=1, class_w=None, data_transform=1):
  lbl_data_dir = data_dir+"annot"
  data_dir = pathlib.Path(data_dir) 
  lbl_data_dir = pathlib.Path(lbl_data_dir)
  lbl_file_names = [f.name for f in pathlib.Path(lbl_data_dir).glob("*.png")]
  #(x_ds, y_ds) = [ (str(data_dir/fname), str(lbl_data_dir/fname) ) for  fname in lbl_file_names]
  x_ds = [ str(data_dir/fname) for  fname in lbl_file_names]
  y_ds = [ str(lbl_data_dir/fname)  for  fname in lbl_file_names]
  
#  for x, y in zip(x_ds,y_ds):
#      print (x, y)
  
#  for x, y in zip(x_ds,y_ds):
#    f = pathlib.Path(x)
#    assert f.is_file()
#    assert pathlib.Path(x).is_file()
#    print(x, y)
    
  x_ds = tf.data.Dataset.list_files(x_ds,seed=0)
  y_ds = tf.data.Dataset.list_files(y_ds,seed=0)
  #x_ds = tf.data.Dataset.list_files(str(sorted(pathlib.Path(data_dir).glob("*.png"))))
  
  #lbl_data_dir = pathlib.Path(lbl_data_dir)
  #y_ds = tf.data.Dataset.list_files((str(lbl_data_dir/'*.png')))
  
  process_ds_path = lambda x : process_img_from_path(x, im_w=im_w,im_h=im_h,chn=3,dtype=tf.float32)
  x_ds = x_ds.map(process_ds_path)#, num_parallel_calls=AUTOTUNE)
  
  process_ann_ds_path = lambda x : process_img_from_path(x, im_w=im_w,im_h=im_h,chn=1,dtype=tf.uint8, ns=num_classes,reshape=reshape)
  y_ds = y_ds.map(process_ann_ds_path) #, num_parallel_calls=AUTOTUNE)
  
#  lbls_waights = lambda y : y2w (y,class_w)
#  w_ds = y_ds.map(lbls_waights, num_parallel_calls=AUTOTUNE)
  
  #y_ds.map(lambda y : to_categorical(y, num_classes=num_classes) , num_parallel_calls=AUTOTUNE)train_ds.take(1)
  #y_ds.map(lambda y : tf.reshape(y,(-1,num_classes)))
  #y_ds = to_categorical(y_ds,num_classes=num_classes)
  labeled_ds = tf.data.Dataset.zip((x_ds , y_ds))
  if data_transform is not None :
      labeled_ds = labeled_ds.map(flip_data, num_parallel_calls=AUTOTUNE)
  # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
  return labeled_ds


def create_poisson_rand_dataset(data_dir,im_w=None,im_h=None, num_classes=12,  reshape=1, class_w=None, data_transform=1):
  lbl_data_dir = data_dir+"annot"
  data_dir = pathlib.Path(data_dir) 
  lbl_data_dir = pathlib.Path(lbl_data_dir)
  lbl_file_names = [f.name for f in pathlib.Path(lbl_data_dir).glob("*.png")]

  x_ds = [ str(data_dir/fname) for  fname in lbl_file_names]
  y_ds = [ str(lbl_data_dir/fname)  for  fname in lbl_file_names]
  
  x_ds = tf.data.Dataset.list_files(x_ds,seed=0)
  y_ds = tf.data.Dataset.list_files(y_ds,seed=0)
  
  process_ds_path     = lambda x : process_img_from_path(x, im_w=im_w,im_h=im_h,chn=3,dtype=tf.float32)
  x_ds = x_ds.map(process_ds_path)#, num_parallel_calls=AUTOTUNE)
  
  process_ann_ds_path = lambda x : process_img_from_path(x, im_w=im_w,im_h=im_h,chn=3,dtype=tf.float32)
  y_ds = y_ds.map(process_ann_ds_path) #, num_parallel_calls=AUTOTUNE)

  labeled_ds = tf.data.Dataset.zip((x_ds , y_ds))
  
  return labeled_ds


if __name__ == '__main__':
  
  
  # --------
    dw = 64
    nc=12
    
    data_dir = "dataset/train"                
    train_ds = create_dataset(data_dir,im_w=dw,im_h=dw, num_classes=nc)
    
    data_dir = "dataset/val"                
    train_ds = create_dataset(data_dir,im_w=dw,im_h=dw, num_classes=nc,reshape=None)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
#    data_dir = "dataset/val"
#    val_ds = create_dataset(data_dir,im_w=dw,im_h=dw, num_classes=nc)
#    print("-------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
#    #train_ds.concatenate(val_ds)
#    #train_ds = prepare_for_training(train_ds, batch_size=1, cache=None, shuffle_buffer_size=1000)
    train_ds = pautoencmodelrepare_for_training(train_ds,batch_size=12,cache=None,shuffle_buffer_size=1000) 
    for im ,lbl in train_ds.take(1):
      print("Image shape: ", im.numpy().shape)
      print("Label shape: ", lbl.numpy().shape)
      
      
      
      
      
 # --------  CoCo ----
    val_ds = create_coco_dataset() 
    print(val_ds.take(1))
    train_ds = create_coco_dataset(dataType='train2017')
    for im ,lbl in train_ds.take(1):
      print("Image, Image shape: ", im, im.numpy().shape)
      print("Lable, Label shape: ", lbl, lbl.numpy().shape)
      show_img_lbl(im,lbl)
 #------
    test_ds = create_coco_test_set()
    print(test_ds.take(1))
    im = []
    lbl = []
    lbl3 = []
    for im ,lbl in test_ds.take(5):
        print("Image, Image shape: ",  im.numpy().shape)
        print("Lable, Label shape: ",  lbl.numpy().shape)
        lbl3 =lbl
        lbl2 = tf.math.argmax(lbl,axis=-1).numpy()
        plt.figure(figsize=(10,10))
        ax = plt.subplot(1,2,1)
        plt.imshow(im)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(lbl2)
        plt.axis('off')
        plt.show()
 #------------------------------   
#  data_dir = "dataset/train"
#  #data_dir = tf.keras.utils.get_file(origin='url', fname='flower_photos', untar=True)
#  data_dir = pathlib.Path(data_dir)
#  
#  image_count = len(list(dae_batch.numpy()ta_dir.glob('*.png')))
#
#  CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
#
#  roses = list(data_dir.glob('roses/*'))
#  train_ds = create_dataset(data_dir,im_w=dw,im_h=dw, num_classes=nc)
#  for image_path in roses[:3]:
#    display.display(Image.open(str(image_path)))
#    
#  # The 1./255 is to convert from uint8 to float32 in range [0,1].
#  image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#
#
#  BATCH_SIZE = 32
#  IMG_HEIGHT = 224
#  IMG_WIDTH = 224
#  STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
#  
#  train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
#                                                       batch_size=BATCH_SIZE,
#                                                       shuffle=True,
#                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                       classes = list(CLASS_NAMES))
#  
#  image_batch, label_batch = next(train_data_gen)
#  show_batch(image_batch, label_batch)
  

#  data_dir = "dataset/train"
#  data_dir = pathlib.Path(data_dir)
#  list_ds = tf.data.Dataset.list_files(str(data_dir/'*.png'))
#  for f in list_ds.take(5):
#    print(f.numpy())
#   
#  # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
#  labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#  
#  for image in labeled_ds.take(1):
#    print("Image shape: ", image.numpy().shape)
#   # print("Label: ", label.numpy())
#
#  train_ds = prepare_for_training(labeled_ds)
#  image_batch = next(iter(train_ds))
#  show_batch(image_batch.numpy())
#  
#  data_dir = "dataset/train"
#  train_ds = create_dataset(data_dir,im_w=None,im_h=None,chn=3,dtype=tf.float32)
#  data_dir = "dataset/trainannot"
#  train_lbl_ds = create_dataset(data_dir,im_w=None,im_h=None,chn=1,dtype=tf.uint8)
#  labeled_ds = tf.data.Dataset.zip((train_ds , train_lbl_ds))
#  #labeled_ds = train_ds.zip(train_lbl_ds)
#  #labeled_ds = tf.data.Dataset.from_tensor_slices([train_ds,train_lbl_ds])
#  for im ,lbl in labeled_ds.take(2):
#    print("Image shape: ", im.numpy().shape)
#    print("Label: ", lbl.numpy().shape)
#    
#  #labeled_ds = 
#
## A one-shot iterator automatically initializes itself on first use.
##iterator = dset.make_one_shot_iterator()
#
#  # The return value of get_next() matches the dataset element type.
#  images, labels = next(iter(labeled_ds))
#  show_img_lbl(images,labels)



    
















    