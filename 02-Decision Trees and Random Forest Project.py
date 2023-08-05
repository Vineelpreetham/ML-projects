#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('loan_data.csv')


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.head()


# In[10]:


sns.set_style('whitegrid')


# In[17]:


plt.figure(figsize=(10,8))
df[df['credit.policy']==1]['fico'].hist(alpha =0.5,color ='blue',bins = 30,label = 'Credi=1')
df[df['credit.policy']==0]['fico'].hist(alpha =0.5,color ='red',bins = 30,label = 'Credit.Policy=0')  
plt.legend()
plt.xlabel("FICO")
                                              


# ** Create a similar figure, except this time select by the not.fully.paid column.**

# In[18]:


plt.figure(figsize=(10,8))
df[df['not.fully.paid']==1]['fico'].hist(alpha =0.5,color ='blue',bins = 30,label = 'not.fully.paid=1')
df[df['not.fully.paid']==0]['fico'].hist(alpha =0.5,color ='red',bins = 30,label = 'not.fully.paid=0')  
plt.legend()
plt.xlabel("FICO")


# In[25]:


plt.figure(figsize=(11,8))
sns.countplot(x = 'purpose' ,data = df , hue =  'not.fully.paid')


# In[ ]:





# In[28]:


sns.jointplot(x = 'fico', y = 'int.rate',data = df,color = 'purple')


# In[30]:


sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# In[31]:


df.info()


# In[32]:


cat_feats = ['purpose']


# In[33]:


final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)


# In[34]:


final_data.info()


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X = final_data.drop('not.fully.paid',axis = 1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ## Training a Decision Tree Model
# 
# Let's start by training a single decision tree first!
# 
# ** Import DecisionTreeClassifier**

# In[37]:


from sklearn.tree import DecisionTreeClassifier


# In[38]:


dtree = DecisionTreeClassifier()


# In[39]:


dtree.fit(X_train,y_train)


# In[40]:


pred = dtree.predict(X_test)


# In[41]:


from sklearn.metrics import classification_report,confusion_matrix


# In[42]:


print(confusion_matrix(pred,y_test))


# In[44]:


print(classification_report(pred,y_test))


# In[45]:


from sklearn.ensemble import RandomForestClassifier


# In[50]:


rfc = RandomForestClassifier(n_estimators=600)


# In[51]:


rfc.fit(X_train,y_train)


# In[52]:


pred_rfc = rfc.predict(X_test)


# In[53]:


print(confusion_matrix(pred_rfc,y_test))


# In[54]:


print(classification_report(pred_rfc,y_test))


# Decision TREE
