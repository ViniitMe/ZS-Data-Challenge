import pandas as pd # for data manipulation and importing files
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


#Loading dataset
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

#Dataset information retrieval
print(train.shape)
print(train.info())
print(train.isnull().sum())
print(test.shape)
print(test.info())
print(test.isnull().sum())

#Splitting into train and test test
x_train=train.drop(['bestSoldierPerc'],1)
y_train=train['bestSoldierPerc']
x_test=test

#Preprocessing
#Handling missing values
im=SimpleImputer()
x_train=im.fit_transform(x_train)

#Feature Scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Training

#Model1
# reg=LinearRegression()
# reg.fit(x_train,y_train)
# y_pred=reg.predict(x_test)
# y_pred=pd.DataFrame(y_pred)

#Model2
regressor=RandomForestRegressor(n_estimators=20)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred=pd.DataFrame(y_pred)


re=pd.read_csv("sample_submission.csv")
re=re.drop(['bestSoldierPerc'],1)
result=pd.concat([re,y_pred],axis=1)
result.to_csv("result1.csv",index=False)