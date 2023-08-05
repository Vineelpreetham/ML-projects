#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('Ecommerce Customers')


# In[4]:


df.head()


# In[23]:


df.columns


# In[7]:


df.describe()


# In[8]:


df.info()


# In[13]:


sns.set_palette('GnBu_d')
sns.set_style('whitegrid')


# In[14]:


sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent',data = df)


# In[ ]:





# In[281]:





# In[15]:


sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent',data = df)


# In[17]:


sns.jointplot('Time on App','Length of Membership',kind = 'hex',data =df)


# In[18]:


sns.pairplot(df)


# In[19]:


#length of membership 


# In[21]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data = df)


# In[26]:


y = df['Yearly Amount Spent']


# In[29]:


x = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[24]:


from sklearn.model_selection import train_test_split 


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=101)


# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


lm = LinearRegression()


# In[38]:


lm.fit(x_train,y_train)


# In[40]:


print(lm.coef_)


# In[42]:


predictions = lm.predict(x_test)


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[44]:


plt.scatter(x = y_test,y = predictions)


# In[46]:


from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[50]:


sns.displot((y_test-predictions) ,bins = 50)


# In[51]:


coeffecients = pd.DataFrame(lm.coef_,x.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

