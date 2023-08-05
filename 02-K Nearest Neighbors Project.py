#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('KNN_Project_Data')


# In[3]:


df.head()


# In[5]:


sns.pairplot(data = df,hue = 'TARGET CLASS',palette ='coolwarm')


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scaler = StandardScaler()


# In[8]:


scaler.fit(df.drop("TARGET CLASS",axis = 1))


# In[9]:


scaled_features = scaler.transform(df.drop("TARGET CLASS",axis = 1))


# In[8]:





# In[10]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[11]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier


# In[14]:


kn = KNeighborsClassifier(n_neighbors=1)


# In[21]:


kn.fit(X_train,y_train)


# In[22]:


pred = kn.predict(X_test)


# ** Create a confusion matrix and classification report.**

# In[23]:


from sklearn.metrics import classification_report,confusion_matrix


# In[24]:


print(confusion_matrix(pred,y_test))


# In[26]:


print(classification_report(pred,y_test))


# In[27]:


error_rate = []

for i in range(1,40):
    kn = KNeighborsClassifier(n_neighbors=i)
    kn.fit(X_train,y_train)
    pred_i = kn.predict(X_test)
    error_rate.append(np.mean(pred_i!= y_test))


# In[32]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[39]:


kn = KNeighborsClassifier(n_neighbors=34)
kn.fit(X_train,y_train)
pred = kn.predict(X_test)


print('with k = 35')
print('\n')
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))

