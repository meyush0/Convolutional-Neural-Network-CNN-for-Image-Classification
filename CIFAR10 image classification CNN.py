#!/usr/bin/env python
# coding: utf-8

# # CNN-Convolution Neural Network

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets,models,layers
from tensorflow.keras.layers import Dense

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ## Download and prepare the CIFAR10 dataset
# ##### The CIFAR10 dataset contains 60,000 color images in 10 classes, with 6,000 images in each class.
# ##### The dataset is divided into 50,000 training images and 10,000 testing images. The classes are mutually exclusive and there is no overlap between them.

# In[2]:


(train_image,train_label),(test_image,test_label)=datasets.cifar10.load_data()


# In[3]:


#normilizing the pixel value between [0,1]
train_image,test_image = train_image / 255.0 , test_image / 255.0


# In[4]:


(train_image.shape,train_label.shape),(test_image.shape,test_label.shape)


# In[12]:


train_image


# In[13]:


test_image


# ## Verify the data
# ##### To verify that the dataset looks correct, let's plot the first 25 images from the training set and display the class name below each image:

# In[16]:


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_image[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_label[i][0]])
plt.show()


# ## Create the convolutional base
# ##### The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.
# 
# ##### As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size. If you are new to these dimensions, color_channels refers to (R,G,B). In this example, you will configure your CNN to process inputs of shape (32, 32, 3), which is the format of CIFAR images. You can do this by passing the argument input_shape to your first layer.

# In[49]:


model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))     #convensional operation
model.add(layers.MaxPooling2D(2,2))       #max pooling layer
model.add(layers.Conv2D(62,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(62, (3, 3), activation='relu'))


# In[50]:


model.summary()


# ### Add Dense layers on top
# ##### To complete the model, you will feed the last output tensor from the convolutional base (of shape (4, 4, 64)) into one or more Dense layers to perform classification. Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. First, you will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top. CIFAR has 10 output classes, so you use a final Dense layer with 10 outputs.

# In[51]:


model.add(layers.Flatten())
model.add(layers.Dense(units=65,activation='relu'))
model.add(layers.Dense(units=10))


# In[52]:


model.summary()


# ## Compile and train the model

# In[53]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_image, train_label, epochs=10, 
                    validation_data=(test_image, test_label))


# In[ ]:





# ## Evaluate the model

# In[62]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_image,  test_label, verbose=2)


# In[64]:


test_acc*100


# In[ ]:




