#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


a=pd.read_csv('C:/Users/hp/Downloads/train_ctrUa4K.csv')
a


# In[8]:


a.isna().sum()


# In[9]:


data=a.drop(['Loan_ID'],axis=1)
data


# In[10]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
data['Gender']=labelencoder.fit_transform(data['Gender'])
data['Married']=labelencoder.fit_transform(data['Married'])
data['Education']=labelencoder.fit_transform(data['Education'])
data['Self_Employed']=labelencoder.fit_transform(data['Self_Employed'])
data['Property_Area']=labelencoder.fit_transform(data['Property_Area'])
data['Loan_Status']=labelencoder.fit_transform(data['Loan_Status'])
data['Dependents']=labelencoder.fit_transform(data['Dependents'])


# In[11]:


data.head()


# In[12]:


data.isna().sum()


# In[13]:


for i in['LoanAmount','Credit_History','Loan_Amount_Term']:
    data[i]=data[i].fillna(data[i].median())


# In[14]:


data.isna().sum()


# In[15]:


y=data['Loan_Status']
x=data.drop(['Loan_Status'],axis=1)


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(x_train,y_train) 
predictions=model.predict(x_test)
predictions


# In[18]:


y_test


# In[19]:


from sklearn.metrics import mean_squared_error,r2_score
print('the mean squared error is',mean_squared_error(y_test,predictions))
print('the r2 squared error is',r2_score(y_test,predictions))


# In[20]:


data.shape


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)


# In[22]:


from sklearn.linear_model import LogisticRegression
logit_model=LogisticRegression()
logit_model.fit(x_train,y_train)
y_pred=logit_model.predict(x_test)
y_pred



# In[23]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
print('Accuracy is',accuracy_score(y_test,y_pred))
print('Precision is',precision_score(y_test,y_pred))
print('Recall is',recall_score(y_test,y_pred))
print('f1 score is',f1_score(y_test,y_pred))


# In[24]:


confusion_matrix(y_test,y_pred)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
metric_k=[]
neighbors=np.arange(3,15)

for k in neighbors:
    classifier=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    metric_k.append(acc)


# In[26]:


plt.plot(neighbors,metric_k,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.grid()


# In[27]:


classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# In[28]:


y_pred.shape


# In[29]:


print('Accuracy is',accuracy_score(y_test,y_pred))
print('Precision is',precision_score(y_test,y_pred))
print('Recall is',recall_score(y_test,y_pred))
print('f1 score is',f1_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)


# In[30]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# In[31]:


logit_model=LogisticRegression()
logit_model.fit(x_train,y_train)
y_pred=logit_model.predict(x_test)


# In[32]:


y_pred.shape


# In[33]:


print('Accuracy is',accuracy_score(y_test,y_pred))
print('Precision is',precision_score(y_test,y_pred))
print('Recall is',recall_score(y_test,y_pred))
print('f1 score is',f1_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)


# In[34]:


from sklearn.neighbors import KNeighborsClassifier
metric_k=[]
neighbors=np.arange(3,15)

for k in neighbors:
    classifier=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    metric_k.append(acc)


# In[35]:


plt.plot(neighbors,metric_k,'o-')# graph shows for every k value we get a same accuracy level
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.grid()


# In[36]:


classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# In[37]:


print('Accuracy is',accuracy_score(y_test,y_pred))
print('Precision is',precision_score(y_test,y_pred))
print('Recall is',recall_score(y_test,y_pred))
print('f1 score is',f1_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)


# In[38]:


from sklearn.svm import SVC
svmclf=SVC(kernel='linear')
svmclf.fit(x_train,y_train)


# In[39]:


y_pred=svmclf.predict(x_test)


# In[40]:


from sklearn.metrics import accuracy_score,confusion_matrix
print('Accuracy is:',accuracy_score(y_test,y_pred))


# In[41]:


print(confusion_matrix(y_test,y_pred))


# In[42]:


#decision tree


# In[43]:


from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)
y_pred=dt_clf.predict(x_test)


# In[44]:


print('Accuracy is:',accuracy_score(y_test,y_pred))


# In[45]:


print(confusion_matrix(y_test,y_pred))


# In[46]:


#Random forest


# In[47]:


from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier()
rf_clf.fit(x_train,y_train)


# In[48]:


y_pred=rf_clf.predict(x_test)


# In[49]:


print('Accuracy is:',accuracy_score(y_test,y_pred))


# In[50]:


print(confusion_matrix(y_test,y_pred))


# In[51]:


#testdata


# In[52]:


testdata=pd.read_csv('C:/Users/hp/Downloads/test_lAUu6dG (1).csv')
testdata


# In[53]:


testdata.isna().sum()


# In[54]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
testdata['Gender']=labelencoder.fit_transform(testdata['Gender'])
testdata['Married']=labelencoder.fit_transform(testdata['Married'])
testdata['Education']=labelencoder.fit_transform(testdata['Education'])
testdata['Self_Employed']=labelencoder.fit_transform(testdata['Self_Employed'])
testdata['Property_Area']=labelencoder.fit_transform(testdata['Property_Area'])
testdata['Dependents']=labelencoder.fit_transform(testdata['Dependents'])


# In[55]:


testdata.drop('Loan_ID',axis=1).head()


# In[56]:


testdata.shape


# In[57]:


testdata.isna().sum()


# In[58]:


for i in['LoanAmount','Loan_Amount_Term','Credit_History']:
    testdata[i]=testdata[i].fillna(testdata[i].median())


# In[59]:


testdata.shape


# In[60]:


testdata.isna().sum()


# In[61]:


x_test=testdata.drop(['Loan_ID'],axis=1)
x_test


# In[62]:


y_pred=rf_clf.predict(x_test)
y_pred


# In[63]:


aaa=y_pred
bbb=[]
for i in range(len(aaa)):
#     print(aaa[i])
    if aaa[i] == 1:
        bbb.append('Y')
    else:
        bbb.append('N')
ccc = np.array(bbb)
ccc


# In[64]:


y_pred1=pd.DataFrame(ccc)
y_pred1


# In[95]:


sampledata=pd.read_csv('C:/Users/hp/Downloads/sample_submission_49d68Cx (1).csv')
sampledata


# In[96]:


sampledata['Loan_Status']=y_pred1
sampledata


# In[97]:


sampledata.to_csv('C:/Users/hp/Downloads/Assesment3.csv', index=False)


# In[ ]:




