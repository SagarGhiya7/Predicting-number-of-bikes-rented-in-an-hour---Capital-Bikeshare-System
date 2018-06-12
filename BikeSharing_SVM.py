
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
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


# # Implementing Support Vector Regression with tuning C and gamma

# In[6]:


svr = SVR(kernel = 'linear')


# In[13]:


Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(estimator = svr, param_grid = param_grid, cv=5)
grid_search.fit(x_trn, y_trn)


# In[15]:


grid_search.best_params_


# In[16]:


final_model_svr = grid_search.best_estimator_
final_model_svr


# In[17]:


# Evaluating model with Root Mean Squared Error
def evaluate(model, features, labels):
    predictions = model.predict(features)
    mse = mean_squared_error(labels,predictions)
    return np.sqrt(mse)
    


# In[18]:


train_error = evaluate(final_model_svr, x_trn, y_trn)
print("Train error is : " , train_error)


# In[19]:


test_error = evaluate(final_model_svr, x_tst, y_tst)
print("Test error is : ", test_error)

