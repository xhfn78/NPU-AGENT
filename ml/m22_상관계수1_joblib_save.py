import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1데이터
datasets = load_iris()
print(datasets.feature_names)
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets['data']
y= datasets.target



df = pd.DataFrame(x,columns=datasets.feature_names)
print(df)

df['Target'] = y
print(df)

path= './_save/pandas_pkl/'
import pickle
pickle.dump(df,open(path +'m22_pandas_save.pkl', 'wb')) #pickle 저장하기

path= './_save/pandas_joblib/'
import joblib
joblib.dump(df,path +'m22_pandas_save.joblib') #joblib 저장하기

#    sepal length (cm)  sepal width (cm)  ...  petal width (cm)  Target
# 0                  5.1               3.5  ...               0.2       0
# 1                  4.9               3.0  ...               0.2       0
# 2                  4.7               3.2  ...               0.2       0
# 3                  4.6               3.1  ...               0.2       0
# 4                  5.0               3.6  ...               0.2       0
# ..                 ...               ...  ...               ...     ...
# 145                6.7               3.0  ...               2.3       2
# 146                6.3               2.5  ...               1.9       2
# 147                6.5               3.0  ...               2.0       2
# 148                6.2               3.4  ...               2.3       2
# 149                5.9               3.0  ...               1.8       2

# [150 rows x 5 columns]