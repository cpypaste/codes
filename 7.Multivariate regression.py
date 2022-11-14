import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
dataset= load_iris()
print(dataset)

data = pd.read_csv("C:\\Users\\online\\Desktop\\codes\\iris.csv")
data

data['Species'].unique()

data['Species'] = data['Species'].replace({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})
data

X_train, X_test, y_train, y_test = train_test_split(data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],data['Species'], test_size=0.3)

from sklearn import linear_model
mymodel = linear_model.LogisticRegression(max_iter=120)
mymodel.fit(X_train,y_train)

print(mymodel.predict(X_test))

print(mymodel.score(X_test, y_test))

predicted_output = mymodel.predict(X_test)
predicted_output




from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_output)
print(cm)

import seaborn as sn
from matplotlib import pyplot as plt
plt.figure(figsize = (5,4))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted Value')
plt.ylabel('Truth or Actual Value')
plt.show()