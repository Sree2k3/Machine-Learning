#!/usr/bin/env python
# coding: utf-8

# # Cost Function

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('deeplearning.mplstyle')


# In[10]:


x_train = np.array([1.0, 2.0])        #( Size in 1000 square feet)
y_train = np.array([300.0, 500.0])    #(price in 1000s of dollars)


# # Computing cost

# In[11]:


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
               """
    #number of training examples
    m = x.shape[0]
    
    cost_sum=0
    for i in range(m):
        f_wb = w*x[i]+b
        cost = (f_wb - y[i])**2
        cost_sum = cost_sum +cost
    total_cost = (1/(2*m))* cost_sum
    
    return total_cost


# # Cost function Intuition

# In[7]:


plt_intuition(x_train,y_train)


# # large dataset for visualization

# In[12]:


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])


# In[13]:


plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)


# In[14]:


soup_bowl()


# In[ ]:




