import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
iris=load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

iris_petal=iris.data[:,[2,3]]

iris_df["label"] = iris.target

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.pairplot(iris_df, hue='label', palette='bright')
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=7)

dt_model=DecisionTreeClassifier(random_state=32)   #결정트리 모델
dt_model.fit(X_train,Y_train)

y_pred=dt_model.predict(X_test)
y_pred

from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))

knn_model = KNeighborsClassifier(n_neighbors=5)     #knn 모델
knn_model.fit(X_train,Y_train)

y_pred1=dt_model.predict(X_test)
y_pred1

from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred1))
