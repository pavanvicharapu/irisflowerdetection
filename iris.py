#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[6]:


df = pd.read_excel(r'C:\Users\91994\Downloads\IRIS.xlsx')
print(df.head())


# In[8]:


df.describe()


# In[9]:


df.info()


# In[11]:


df['species'].value_counts()


# In[12]:


df.isnull().sum()


# In[14]:


df['sepal_length'].hist()


# In[15]:


df['petal_width'].hist()


# In[18]:


df['petal_length'].hist()


# In[19]:


colors=['orange','blue','green']
species=['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[24]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label=species[i])
plt.xlabel("sepallenth")
plt.ylabel("sepalwidth")
plt.legend()


# In[25]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel("petallenth")  
plt.ylabel("petalwidth")
plt.legend()


# In[26]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['petal_length'],c=colors[i],label=species[i])
plt.xlabel("sepallenth")  
plt.ylabel("petalwidth")
plt.legend()


# In[27]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_width'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel("sepalwidth")  
plt.ylabel("petalwidth")
plt.legend()


# In[28]:


df.corr()


# In[31]:


corr=df.corr()
fig, ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')


# In[32]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()


# In[33]:


df['species']=le.fit_transform(df['species'])
df.head()


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Assuming you've already loaded your DataFrame 'df'

X = df.drop(columns=['species'])
Y = df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
model = LogisticRegression()
model.fit(x_train, y_train)


# In[42]:


print("Accuracy:",model.score(x_test,y_test))


# In[37]:





# In[43]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:




