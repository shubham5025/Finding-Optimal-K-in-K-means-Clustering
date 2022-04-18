import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#importing the Dataset
dataset=pd.read_csv('t.csv',header=None)
x=dataset.iloc[:,:5].values
y=dataset.iloc[:,-1].values

#Oversampling the Data
from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler(random_state=0)
x,y,=os.fit_resample(x,y)

#Splitting the Dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

#Ensemble method based Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=10, 
                                   criterion='entropy', 
                                   random_state=0, 
                                   max_features=1, 
                                   min_samples_leaf=1)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

#Result Analysis
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators':[1,5,6,7,8,9,10,11,12,13,14,15,16],
            'criterion':['gini','entropy'],
            'min_samples_leaf':[1,2,3,4,5], 
            'max_features':[1,2,3,4,5]}
grid=GridSearchCV(classifier,param_grid,refit=True)
grid.fit(x_train,y_train)
print(grid.best_params_)

#Result Analysis on test set
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score
print(classification_report(y_test,y_pred,digits=3))
accuracy_score(y_test, y_pred)
precision_score(y_test,y_pred,average='weighted')
recall_score(y_test, y_pred, average='weighted')


#Using Balanced Bagging classifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Create an instance
classifier1 = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(),
                                sampling_strategy='not majority',
                                replacement=False,
                                random_state=42)

classifier2 = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='not majority',
                                replacement=False,
                                random_state=42)

classifier1.fit(x_train, y_train)
classifier2.fit(x_train, y_train)
y_pred1= classifier1.predict(x_test)
y_pred2= classifier2.predict(x_test)

#Result Analysis on test set
print(classification_report(y_test,y_pred1,digits=3))
print(classification_report(y_test, y_pred2,digits=3))


#importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Initialising Artificial Neural Network
classifier3=Sequential()

#Adding the input layer and first hidden layer
classifier3.add(Dense(units=10,input_dim=5, activation='relu'))

#Adding Second hidden layer
classifier3.add(Dense(units=10, activation='relu'))

#Adding Output Layer
classifier3.add(Dense(units=12,activation='softmax'))

#Compiling ANN
classifier3.compile(optimizer='adam', 
                   loss='sparse_categorical_crossentropy', 
                   metrics=('accuracy'))

#Fitting the ANN to training test
classifier3.fit(x_train,y_train, batch_size=5, epochs=50)

#Predicted the test result
y_pred3=classifier3.predict(x_test)

#Result Analysis on test set
print(classification_report(y_test,y_pred3,digits=3))