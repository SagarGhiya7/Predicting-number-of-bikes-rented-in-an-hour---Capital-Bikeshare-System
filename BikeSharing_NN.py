
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import math
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


os.chdir("C:/Users/Sagar Ghiya/Desktop")


# In[3]:


#Importing data
df = pd.read_csv('hour.csv')
df.head()


# In[4]:


#Dropping columns that are not needed for building model.
df1 = df.drop(['instant', 'dteday', 'casual', 'registered'], axis = 1)
df1.head()


# In[5]:


#Dividing data into train and test
df2 = df1.drop(['cnt'], axis = 1)
y = df1['cnt']
x_trn, x_tst, y_trn, y_tst = train_test_split(df2,y, test_size = 0.3)


# # Training Deep Neural Network

# In[18]:


model_nn = MLPRegressor(hidden_layer_sizes = (30,30,30,30), activation = 'relu', solver = 'sgd', batch_size = 128, max_iter = 1000000, random_state = 42)


# In[19]:


#Initializing hyperparameters
param_grid = {'alpha' : [0.01, 0.1, 1] , 'learning_rate_init':[0.00001, 0.0001, 0.001,0.01] }


# In[ ]:


#Cross validation to find optimalhyperparameters
grid_search = GridSearchCV(estimator = model_nn, param_grid = param_grid, cv = 5, n_jobs = -1)
grid_search.fit(x_trn, y_trn)


# In[ ]:


grid_search.best_params_


# In[ ]:


final_model_nn = grid_search.best_estimator_
final_model_nn


# In[ ]:


# Evaluating model with Root Mean Squared Error
def evaluate(model, features, labels):
    predictions = model.predict(features)
    mse = mean_squared_error(labels,predictions)
    return np.sqrt(mse)


# In[ ]:


train_error = evaluate(final_model_nn, x_trn, y_trn)
print("Train error is : " , train_error)


# In[ ]:


test_error = evaluate(final_model_nn, x_tst, y_tst)
print("Test error is : ", test_error)

