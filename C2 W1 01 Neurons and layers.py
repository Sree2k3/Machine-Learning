#!/usr/bin/env python
# coding: utf-8

# # Neurons and Layers 

# #### Packages

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
plt.style.use('deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# ## Neuron without activation - Regression/Linera Model

# #### DataSet

# In[3]:


X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.legend( fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()


# ## Regression/Linear Model

# In[4]:


linear_layer = tf.keras.layers.Dense(units=1, activation ='linear')


# In[5]:


linear_layer.get_weights()


# In[6]:


a1 =linear_layer(X_train[0].reshape(1,1))
print(a1)


# In[7]:


w,b = linear_layer.get_weights()
print(f"w= {w}, b={b}")


# In[8]:


set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())


# In[9]:


a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,X_train[0].reshape(1,1))+set_b
print(alin)


# In[10]:


prediction_tf = linear_layer(X_train)
prediction_np = np.dot(X_train,set_w)+set_b


# In[11]:


plt_linear(X_train, Y_train, prediction_tf,prediction_np)


# ## Neuron with Sigmoid activation

# ### DataSet

# In[12]:


X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix


# In[13]:


pos =Y_train ==1
neg =Y_train ==1
X_train[pos]


# In[14]:


pos = Y_train == 1
neg = Y_train == 0

fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors=dlc["dlblue"],lw=3)

ax.set_ylim(-0.08,1.1)
ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_title('one variable plot')
ax.legend(fontsize=12)
plt.show()


# ## Logistic Neuron

# In[15]:


model = Sequential(
         [
             tf.keras.layers.Dense(1, input_dim=1, activation = 'sigmoid', name='L1')
         ]
)


# In[16]:


model.summary()


# In[18]:


logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)


# In[21]:


set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())


# In[24]:


a1 = model.predict(X_train[0].reshape(1,1))
print(a1)
alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1))+set_b)
print(alog)


# In[25]:


plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)


# In[ ]:




