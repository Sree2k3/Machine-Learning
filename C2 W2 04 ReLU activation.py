#!/usr/bin/env python
# coding: utf-8

# # ReLU activation
# 

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import linear, relu, sigmoid
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from autils import plt_act_trio
from lab_utils_relu import *
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


# ## ReLu Activation
# 

# In[7]:


plt_act_trio()


# In[8]:


_= plt_relu_ex()


# In[ ]:




