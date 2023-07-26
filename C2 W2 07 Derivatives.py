#!/usr/bin/env python
# coding: utf-8

# # Derivatives

# In[1]:


from sympy import symbols, diff


# In[2]:


J = (3)**2
J_epsilon = (3 + 0.001)**2
k = (J_epsilon - J)/0.001    # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")


# In[3]:


J = (3)**2
J_epsilon = (3 + 0.000000001)**2
k = (J_epsilon - J)/0.000000001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# ##  $J = w^2$

# In[4]:


J, w = symbols('J, w')


# In[5]:


J=w**2
J


# In[6]:


dJ_dw = diff(J,w)
dJ_dw


# In[7]:


dJ_dw.subs([(w,2)])    # derivative at the point w = 2


# In[8]:


dJ_dw.subs([(w,3)])    # derivative at the point w = 3


# In[9]:


dJ_dw.subs([(w,-3)])    # derivative at the point w = -3


# ## $J=2w$

# In[10]:


w, J = symbols('w, J')


# In[11]:


J = 2 * w
J


# In[12]:


dJ_dw = diff(J,w)
dJ_dw


# In[13]:


dJ_dw.subs([(w,-3)])    # derivative at the point w = -3


# In[14]:


J = 2*3
J_epsilon = 2*(3 + 0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# ## $J=w^3$

# In[15]:


J, w = symbols('J, w')


# In[16]:


J=w**3
J


# In[17]:


dJ_dw = diff(J,w)
dJ_dw


# In[18]:


dJ_dw.subs([(w,2)])   # derivative at the point w=2


# In[19]:


J = (2)**3
J_epsilon = (2+0.001)**3
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# ## $J= \frac{1}{w}$

# In[20]:


J, w = symbols('J, w')


# In[21]:


J= 1/w
J


# In[22]:


dJ_dw = diff(J,w)
dJ_dw


# In[23]:


dJ_dw.subs([(w,2)])


# In[24]:


J = 1/2
J_epsilon = 1/(2+0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# ## $J= \frac{1}{w^2}$

# In[30]:


J, w = symbols('J, w')


# In[34]:


J= 1/w**2
J


# In[35]:


dJ_dw = diff(J,w)
dJ_dw


# In[33]:


dJ_dw.subs([(w,4)])


# In[36]:


J = 1/4**2
J_epsilon = 1/(4+0.001)**2
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# In[ ]:




