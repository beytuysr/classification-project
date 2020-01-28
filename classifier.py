# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:26:12 2019
classification project
@author: Beytu
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2.Data preprocessing

#2.1.import data
veriler = pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4].values
y=veriler.iloc[:,4:].values 

#split the data train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#data normalization
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

#machine learning part using different algorithm

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print('prediction:', y_pred)
print('test:', y_test)

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
print('confusion matrix of logistic regression')
print(cm)

#2.KNN
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print('confusion matrix of KNN')
print(cm)

#3.SVC (SVM classifier)
from sklearn.svm import SVC
svc=SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print('confusion matrix of SVC')
print(cm)

#4.Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred=gnb.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print('confusion matrix of GNB')
print(cm)

# 5. Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('confusion matrix of DTC')
print(cm)

# 6.Random Forest 
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')

rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('confusion matrix of RFC')
print(cm)

#roc curve
y_proba=rfc.predict_proba(X_test)
print('test:', y_test)
print('y proba:',y_proba[:,0])

from sklearn.metrics import roc_curve
mrc=roc_curve(y_test,y_proba[:,0],pos_label='e')
fpr,tpr,thold=mrc
print('false positive rate:', fpr)
print('true positive rate:',tpr)
print('thres holds:', thold)
