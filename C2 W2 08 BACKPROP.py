#!/usr/bin/env python
# coding: utf-8

# # Backpropagation using a computation graph

# In[3]:


from sympy import *
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets
from lab_utils_backprop import *


# ### Computation graph

# In[19]:


plt.close("all")
plt_network(config_nw0, "./images/C2_W2_BP_network0.PNG")


# ### Forward Propagation

# In[7]:


w = 3
a = 2+3*w
J = a**2
print(f"a = {a}, J = {J}")


# ### Backprop

# In[8]:


a_epsilon = a + 0.001       # a epsilon
J_epsilon = a_epsilon**2    # J_epsilon
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")


# In[9]:


sw,sJ,sa = symbols('w,J,a')
sJ = sa**2
sJ


# In[10]:


sJ.subs([(sa,a)])


# In[11]:


dJ_da = diff(sJ, sa)
dJ_da


# ### Arithmetically

# In[12]:


w_epsilon = w + 0.001       # a  plus a small value, epsilon
a_epsilon = 2 + 3*w_epsilon
k = (a_epsilon - a)/0.001   # difference divided by epsilon
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")


# In[13]:


sa = 2 + 3*sw
sa


# In[14]:


da_dw = diff(sa,sw)
da_dw


# In[15]:


dJ_dw = da_dw * dJ_da
dJ_dw


# In[16]:


w_epsilon = w + 0.001
a_epsilon = 2 + 3*w_epsilon
J_epsilon = a_epsilon**2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# ### Computation Graph of a Simple Neural Network

# In[18]:


plt.close("all")
plt_network(config_nw1, "./images/C2_W2_BP_network1.PNG")


# In[20]:


# Inputs and parameters
x = 2
w = -2
b = 8
y = 1
# calculate per step values   
c = w * x
a = c + b
d = a - y
J = d**2/2
print(f"J={J}, d={d}, a={a}, c={c}")


# In[21]:


d_epsilon = d + 0.001
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dd ~= k = {k} ")


# In[22]:


sx,sw,sb,sy,sJ = symbols('x,w,b,y,J')
sa, sc, sd = symbols('a,c,d')
sJ = sd**2/2
sJ


# In[23]:


sJ.subs([(sd,d)])


# In[24]:


dJ_dd = diff(sJ, sd)
dJ_dd


# In[25]:


a_epsilon = a + 0.001         # a  plus a small value
d_epsilon = a_epsilon - y
k = (d_epsilon - d)/0.001   # difference divided by epsilon
print(f"d = {d}, d_epsilon = {d_epsilon}, dd_da ~= k = {k} ")


# In[26]:


sd = sa - sy
sd


# In[27]:


dd_da = diff(sd,sa)
dd_da


# In[28]:


dJ_da = dd_da * dJ_dd
dJ_da


# In[29]:


a_epsilon = a + 0.001
d_epsilon = a_epsilon - y
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")


# In[30]:


# calculate the local derivatives da_dc, da_db
sa = sc + sb
sa


# In[31]:


da_dc = diff(sa,sc)
da_db = diff(sa,sb)
print(da_dc, da_db)


# In[32]:


dJ_dc = da_dc * dJ_da
dJ_db = da_db * dJ_da
print(f"dJ_dc = {dJ_dc},  dJ_db = {dJ_db}")


# In[33]:


# calculate the local derivative
sc = sw * sx
sc


# In[34]:


dc_dw = diff(sc,sw)
dc_dw


# In[35]:


dJ_dw = dc_dw * dJ_dc
dJ_dw


# In[36]:


print(f"dJ_dw = {dJ_dw.subs([(sd,d),(sx,x)])}")


# In[37]:


J_epsilon = ((w+0.001)*x+b - y)**2/2
k = (J_epsilon - J)/0.001  
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# In[ ]:




