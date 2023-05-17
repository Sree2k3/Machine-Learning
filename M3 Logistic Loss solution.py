#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression, Logistic Loss

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('deeplearning.mplstyle')


# In[2]:


soup_bowl()


# In[3]:


x_train = np.array([0.,1,2,3,4,5],dtype=np.longdouble)
y_train = np.array([0,0,0,1,1,1],dtype=np.longdouble)
plt_simple_example(x_train,y_train)


# In[4]:


plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()


# In[5]:


plt_two_logistic_loss_curves()


# In[7]:


plt.close('all')
cst = plt_logistic_cost(x_train,y_train)


# In[ ]:




