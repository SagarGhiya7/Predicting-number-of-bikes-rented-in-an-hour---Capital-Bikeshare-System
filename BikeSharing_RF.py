
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[20]:


os.chdir("C:/Users/Sagar Ghiya/Desktop")


# In[21]:


#Importing data
df = pd.read_csv('hour.csv')


# In[22]:


df.head()


# # Dropping columns not needed for fitting model.

# In[23]:


df1 = df.drop(['instant', 'dteday', 'casual', 'registered'], axis = 1)


# In[24]:


df1.head()


# Checking for missing values. No missing values in the dataset

# In[25]:


df1.isnull().sum()


# In[26]:


df1.describe()


# In[27]:


df1.dtypes


# In[28]:


#Plotting date vs count
plt.plot(df['dteday'], df['cnt'])
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Date vs Count')


# In[29]:


#Count vs Hour
plt.scatter(df1['cnt'],df1['hr'])
plt.xlabel('Count')
plt.ylabel('Hour')
plt.title('Count vs Hour')


# In[30]:


#Checking distribution of key features with histogram
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                   figsize=(8,4))

ax1.hist(df1['season'])
ax2.hist(df1['mnth'])
ax3.hist(df1['hr'])
ax4.hist(df1['holiday'])


# In[31]:


df1.shape


# In[32]:


#Dividing into train and test
df2 = df1.drop(['cnt'], axis = 1)
y = df1['cnt']
x_trn, x_tst, y_trn, y_tst = train_test_split(df2,y, test_size = 0.3)


# # Hyperparameter Tuning
# Tweaking number of trees and maximum features to be used to get best possible accuracy.

# In[ ]:


param_grid = {'max_features': [6,8,10,12] , 'n_estimators': [10,30,50,100,500,1000,2500,5000] }


# In[34]:


rf = RandomForestRegressor()


# In[35]:


grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5)


# In[36]:


grid_search.fit(x_trn, y_trn)


# In[37]:


#Best Combination. n_estimators give almost same error when increased from 1000 to 5000. So no point in increasing it futher.
grid_search.best_params_


# In[38]:


final_model_rf = grid_search.best_estimator_
final_model_rf


# In[39]:


#Function to evaluate model with Root Mean Squared Error
def evaluate(model, features, labels):
    predictions = model.predict(features)
    mse = mean_squared_error(labels,predictions)
    return np.sqrt(mse)
    


# In[40]:


train_error = evaluate(final_model_rf, x_trn, y_trn)
print("Train error is : " , train_error)


# In[41]:


test_error = evaluate(final_model_rf, x_tst, y_tst)
print("Test error is : ", test_error)


# In[42]:


# Variable importance

importances = list(final_model_rf.feature_importances_)
feature_list = list(x_trn.columns.values)

feature_importances = [(feature, float(round(importance, 3))) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); 
plt.title('Variable Importances');

