
from locale import normalize
from sqlite3 import Row
from statistics import correlation
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn import svm
import seaborn as sns


df = pd.read_csv(r'C:\Users\Admin\Downloads\train.csv')


corr = df.corr()


cols = ['PassengerId','Name','Cabin','Ticket']
df.drop(cols,axis=1,inplace=True)


df['Age'].fillna(df['Age'].mean(),inplace=True)


print(df['Embarked'].dropna())
print(df.shape)

x = pd.get_dummies(df,drop_first=False)
print(x)
print(x.shape)
y = df['Survived']
print(df['Survived'].value_counts())



x.drop('Survived',axis=1,inplace=True)
print(x.shape,y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# Logistic Model

logistic_reg = LogisticRegression()
logistic_reg.fit(x_train,y_train)

logistic_reg_predict =logistic_reg.predict(x_test)

print(y_test,y_test.shape)



# KNN model
# KNN 
from sklearn.neighbors import KNeighborsClassifier

KNN_classfier = KNeighborsClassifier(n_neighbors=5)
KNN_classfier.fit(x_train,y_train)
KNN_pred = KNN_classfier.predict(x_test)

confusion_matrix_KNN = metrics.confusion_matrix(y_test,KNN_pred) 



accuracy_score  = metrics.accuracy_score(y_test,KNN_pred)
print("KNN model accuracy","\n",accuracy_score)

raw_data = (1,38.000000,1,0,71.2500,1,0,0,0,1)
raw_data_as_numpy_array = np.array(raw_data)
raw_data_reshaped = raw_data_as_numpy_array.reshape(1,-1)
print(KNN_classfier.predict(raw_data_reshaped))


# Random forest Model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators= 100, max_features=1.0,max_depth=100,min_samples_split=10,min_samples_leaf=4,random_state=1)

rf.fit(x_train,y_train)
prediction = rf.predict(x_test)
print(prediction.shape)
print(x_test.shape)

raw_data = (1,38.000000,1,0,71.2500,1,0,0,0,1)
raw_data_as_numpy_array = np.array(raw_data)
raw_data_reshaped = raw_data_as_numpy_array.reshape(1,-1)
print(rf.predict(raw_data_reshaped))

confu_matrix = metrics.confusion_matrix(y_test,prediction)
print(confu_matrix)

acc = metrics.accuracy_score(y_test,prediction)
print("Ramdom Forest Model Accuracy:",acc)



# SVM Classfier

from sklearn import svm

svm_classifier = svm.SVC()
svm_classifier.fit(x_train,y_train)
predict =  svm_classifier.predict(x_test)
print(predict)


raw_data = (1,38.000000,1,0,71.2500,1,0,0,0,1)
raw_data_as_numpy_array = np.array(raw_data)
raw_data_reshaped = raw_data_as_numpy_array.reshape(1,-1)
print(svm_classifier.predict(raw_data_reshaped))

confusion_matrix = metrics.confusion_matrix(y_test,predict)
print(confusion_matrix)

accuracy_score = metrics.accuracy_score(y_test,predict)
print("SVM model accuracy: ",accuracy_score)




# Plot learning curve
import  scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test,predict,normalize = True)

skplt.metrics.plot_confusion_matrix(y_test,prediction,normalize = True)

skplt.metrics.plot_confusion_matrix(y_test,KNN_pred,normalize = True)
plt.show()

