import pandas as pd 
import numpy as np 
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

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

model=Sequential()
model.add(Dense(32,input_dim=24,kernel_initializer='normal',activation='relu'))  #input dim=num_column_of orig_dataset-1
model.add(Dense(12,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae','acc'])
train = model.fit(x_train, y_train, epochs=20, batch_size=50,  verbose=1, validation_split=0.3)

y_pred=model.predict(x_test)
y_pred=pd.DataFrame(y_pred)

re=pd.read_csv("sample_submission.csv")
re=re.drop(['bestSoldierPerc'],1)
result=pd.concat([re,y_pred],axis=1)
result.to_csv("resultANN_final.csv",index=False)


# model.add(LeakyReLU(alpha=0.1))
# >>> model.add(BatchNormalization())
# >>> model.add(Dense(64,activation='linear'))
# >>> model.add(LeakyReLU(alpha=0.1))
# >>> model.add(Dense(32,activation='linear'))
# >>> model.add(LeakyReLU(alpha=0.1))
# >>> model.add(Dense(12,activation='linear'))
# >>> model.add(LeakyReLU(alpha=0.1))
# >>> model.add(Dense(12,activation='linear'))
# >>> model.add(LeakyReLU(alpha=0.1))
# >>> model.add(Dense(8,activation='linear'))
# >>> model.add(LeakyReLU(alpha=0.1))
# >>> model.add(Dense(1,activation='linear'))
# >>> model.summary()
