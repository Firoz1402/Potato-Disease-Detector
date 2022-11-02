#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pa
import numpy as np
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[2]:


IMAGE_SIZE = 256;
BATCH_SIZE = 32;
CHANNELS=3
EPOCHS=50


# In[3]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
   "PlantVillage",
    shuffle =True,
    image_size= (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE
    

)


# In[4]:


class_names= dataset.class_names


# In[5]:


len(dataset)


# In[6]:


train_size= 0.8
len (dataset)*train_size


# In[7]:


test_ds= dataset.skip(54)
len(test_ds)


# In[8]:


val_size=0.1;
len(dataset)*val_size


# In[9]:


def get_dataset_partition_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1,shuffle=True, shuffle_size=10000):
    ds_size = len(ds);
    if shuffle:
        ds=ds.shuffle(shuffle_size, seed=12)
    
    train_size= int(train_split*ds_size)
    val_size = int(val_split*ds_size)
    
    train_ds = ds.take(train_size)
    val_ds= ds.skip(train_size).take(val_size)
    test_ds= ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds,test_ds


# In[10]:


train_ds, val_ds, test_ds = get_dataset_partition_tf(dataset)


# In[11]:


len(train_ds)


# In[12]:


len(val_ds)


# In[13]:


len(test_ds)


# In[14]:


train_ds= train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds= val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds= test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[15]:


resize_and_rescale= tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
    
])


# In[16]:


data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
    
])


# In[17]:


input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE, CHANNELS)
n_classes=3


model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64 ,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.build(input_shape= input_shape)


# In[18]:


model.summary()


# In[19]:


model.compile(
    optimizer= 'adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[ ]:


model.fit(
    train_ds,
    epochs= EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)


# In[ ]:




