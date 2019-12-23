#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# In[9]:


wine=pd.read_csv("winequality-red (1).csv")


# In[10]:



#preprocessing 
bins=(2,5.5,8)
group_names=['bad','good']
wine['quality']=pd.cut(wine['quality'],bins=bins,labels=group_names)
    
    


# In[12]:


#scalling
label_quality=LabelEncoder()
y=label_quality.fit_transform(wine['quality'])
x=wine.drop('quality',axis=1)


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[35]:




x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)
clf=MLPClassifier()
clf.fit(x_train,y_train)
    
predict=clf.predict(x_test)
accuracy=print("Accuracy Score:",accuracy_score(y_test,predict))
class_report=print("Classification Report:",classification_report(y_test,predict))
    
    
    


# In[ ]:





# In[ ]:




