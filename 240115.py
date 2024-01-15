'''
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
regr = linear_model.LinearRegression()
X = [[164], [179], [162], [170]]
y = [53, 63, 55, 59]
regr.fit(X, y)
plt.scatter(X, y)
y_pred = regr.predict(X)
plt.plot(X, y_pred, color = 'blue', linewidth = 3)
plt.show()
'''


'''
from sklearn import tree
dt_model = tree.DecisionTreeClassifier() 

parents_height = [[180,165],[175, 160],[180,172],[165,160],[171,152]]
child_height = [3,2,2,1,1] 

dt_model.fit(parents_height, child_height) 

dt_pred=dt_model.predict([[175,153]])
print(dt_pred)
'''


import pandas as pd 

std_df = pd.read_csv('Student_Marks.csv')
# print(std_df)
# print(std_df.describe())
# print(std_df.info())
# print(std_df.head())
# print(std_df.corr())

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax=plt.subplots(ncols=2, figsize=(12,6))
# sns.scatterplot(data=std_df, x='number_courses',y='Marks', ax=ax[0])
# sns.scatterplot(data=std_df, x='time_study',y='Marks', ax=ax[1])
# plt.show()


from sklearn.model_selection import train_test_split
x=std_df.drop('Marks',axis=1)
y=std_df['Marks']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr_model=LinearRegression()

lr_model.fit(X_train,Y_train)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

lr_pred=lr_model.predict(X_test)
print(r2_score(lr_pred, Y_test))
print(mean_squared_error(lr_pred,Y_test))