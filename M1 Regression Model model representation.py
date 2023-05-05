#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# x_train is the input variable(size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train= np.array([1.0,2.0])
y_train= np.array([300.0,500.0])
print(f"x_train ={x_train}")
print(f"y_train ={y_train}")


# # Number of Training Examples m

# In[3]:


# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is :{m}")


# or use len()

# In[4]:


# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is :{m}")


# # Training examples

# In[5]:


i=1 # change this to 1 to see(x^i , y^i)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) =( {x_i}, {y_i})")


# # Plotting the DATA

# In[6]:


# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size(1000 sqft)')
plt.show()


# # Model Function

# In[7]:


w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")


# In[8]:


def compute_model_output(x,w,b):
    """
    
    Computes the prediction of a linear model
    Args:
       x(ndarray (m,)) :DATA, m examples
       w,b (scalar)    : model parameters
    Returns
        y(ndarray (m,)): Target values
        """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i]+b
        
    return f_wb


# # Now let's call the compute_model_output function and plot the output.. 

# In[9]:


tmp_f_wb = compute_model_output(x_train, w,b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our prediction')

# plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

#Set title
plt.title("Housing Prices")
# set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# set the x-axis label
plt.xlabel('Size (1000sqft)')
plt.legend()
plt.show()


# # Set w= 200 & b= 100 in the above code 

# # Prediction

# In[10]:


# Assign the values of w, b, and x_i
w = 200
b = 100
x_i = 1.2

# Calculate the cost of a 1200 sq ft property
cost_1200sqft = w * x_i + b

# Print the cost in thousands of dollars
print("$" + str(int(cost_1200sqft)) + " thousand dollars")

