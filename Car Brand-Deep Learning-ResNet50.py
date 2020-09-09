#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from glob import glob


# In[3]:


IMAGE_SIZE = [224, 224]

train_path = 'D:/Data Science/Car Brand/Datasets/train'
valid_path = 'D:/Data Science/Car Brand/Datasets/test'


# In[4]:


# Import the Resnet50 library as shown below and add preprocessing layer to the front of Resnet50",
# Here we will be using imagenet weights\n"
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[5]:


# don't train existing weights\n",
for layer in resnet.layers:
    layer.trainable = False


# In[6]:


# useful for getting number of output classes
folders = glob('D:/Data Science/Car Brand/Datasets/train/*')
len(folders)


# In[7]:


# our layers - you can add more if you want
x = Flatten()(resnet.output)


# In[8]:


prediction=Dense(len(folders), activation='softmax')(x)


# In[9]:


# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[10]:



# view the structure of the model
model.summary()


# In[11]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[12]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[13]:



# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('D:/Data Science/Car Brand/Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[14]:


test_set = test_datagen.flow_from_directory('D:/Data Science/Car Brand/Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[15]:



# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[16]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[17]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('car_model_resnet50.h5')


# In[18]:


y_pred = model.predict(test_set)


# In[19]:


y_pred


# In[20]:



import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[21]:


y_pred


# In[22]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[23]:


model=load_model('car_model_resnet50.h5')


# In[33]:


img=image.load_img('D:/Data Science/Car Brand/Datasets/Test/lamborghini/11.jpg',target_size=(224,224))


# In[34]:


y=image.img_to_array(img)
y


# In[35]:


y.shape


# In[36]:


import tensorflow as tf


# In[37]:


y=np.expand_dims(y,axis=0)
img_brand=preprocess_input(x)
img_brand.shape


# In[38]:


img_brand=img_brand/255


# In[39]:


preds=model.predict(img_brand)
preds


# In[40]:


a=np.argmax(preds, axis=1)


# In[41]:


a


# In[42]:


if(a==0):
    print('Audi')
elif(a==1):
    print('Lumborghini')
elif(a==2):
    print('Mercedes')


# In[ ]:




