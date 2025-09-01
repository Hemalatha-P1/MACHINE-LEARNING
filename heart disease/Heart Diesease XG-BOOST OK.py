import pandas as pd #for reading dataset
import numpy as np # array handling functions
import xgboost as xgb
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("heart.csv")#reading dataset 
#print(dataset) # printing dataset

x = dataset.iloc[:,:-1].values #locating inputs
y = dataset.iloc[:,-1].values #locating outputs

#printing X and Y
print("x=",x)
print("y=",y)

from sklearn.model_selection import train_test_split # for splitting dataset
x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25 ,random_state = 0)
#printing the spliited dataset
print("x_train=",x_train)
print("x_test=",x_test)
print("y_train=",y_train)
print("y_test=",y_test)

#importing algorithm
model = xgb.XGBClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)

y_pred=model.predict(x_test) #testing model
print("y_pred",y_pred) # predicted output
print("ACCURACY SCORE  = ",accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


a1=int(input("ENTER THE AGE= "))
b1=int(input("ENTER THE SEX= "))
c1=int(input("ENTER THE CP= "))
d1=int(input("ENTER THE TRESTBPS="))
e1=int(input("ENTER THE CHOL= "))
f1=int(input("ENTER THE OLDPEAK= "))
g1=int(input("ENTER THE CA= "))



a = model.predict([[a1,b1,c1,d1,e1,g1]])
print(a)
if a==1:
    print("You're Heart Patient")
elif a==0:
    print("You're not Heart Patient")
print('Predicted new output value: %s' % (a))

    


