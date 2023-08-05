#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


ad = pd.read_csv('advertising.csv')


# In[5]:


ad.head()


# In[7]:


ad.info()


# In[8]:


ad.describe()


# In[9]:


sns.set_style('whitegrid')


# In[ ]:





# In[13]:


sns.histplot(x = 'Age',data = ad,bins = 30)


# In[18]:


sns.jointplot(x = 'Age', y = 'Area Income',data = ad)


# In[20]:


sns.jointplot(x = 'Age',y = 'Daily Time Spent on Site', kind = 'kde',data = ad)


# In[21]:


sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage',data = ad)


# In[22]:


sns.pairplot(data = ad , hue = 'Clicked on Ad')


# In[23]:


from sklearn.model_selection import train_test_split


# In[26]:


X = ad[['Daily Time Spent on Site','Age','Area Income', 'Daily Internet Usage','Male']]
y = ad['Clicked on Ad']


# In[27]:


X_train ,X_test , y_train , y_test = train_test_split(X,y,test_size= 0.33 ,random_state= 42)


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[32]:


logmod = LogisticRegression()
logmod.fit(X_train ,y_train)


# In[33]:


predictions = logmod.predict(X_test)


# In[95]:


from sklean.


# In[96]:




