import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\itzgo\\Desktop\\codes\\Datasets\\data preprocessing.csv",)
df = df.replace(0,np.nan)
df = df.fillna(df.mean())
print(df)
x=np.array(df)[:,:-1]
y=np.array(df)[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=0)
print('x train shape:',x_train.shape)
print('x train shape:',x_test.shape)