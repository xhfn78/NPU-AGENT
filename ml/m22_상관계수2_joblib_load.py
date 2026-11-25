import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1데이터



# print(df)
# path= './_save/pandas_pkl/'
# import pickle
# pickle.dump(df,open(path +'m22_pandas_save.pkl', 'wb')) #pickle 저장하기

path= './_save/pandas_joblib/'
import joblib
df = joblib.load(path+'m22_pandas_save.joblib')
# joblib.dump(df,path +'m22_pandas_save.joblib') #joblib 저장하기
# print(df)


print('===================상관관계 히트맵========================')
print(df.corr())

# ===================상관관계 히트맵========================
#                    sepal length (cm)  sepal width (cm)  ...  petal width (cm)    Target
# sepal length (cm)           1.000000         -0.117570  ...          0.817941  0.782561
# sepal width (cm)           -0.117570          1.000000  ...         -0.366126 -0.426658
# petal length (cm)           0.871754         -0.428440  ...          0.962865  0.949035
# petal width (cm)            0.817941         -0.366126  ...          1.000000  0.956547
# Target                      0.782561         -0.426658  ...          0.956547  1.000000

#항상 기준은 절대값이다 절대값 1 0 -1 리니어로 측정한 값임


import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(),square=True,annot=True,cbar=True)
plt.show()
