import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = SimpleImputer() #디폴트 평균값.

data = pd.DataFrame([[2, np.nan, 6, 8, 10,8],  #x3 빼고 나머지는 전부다 선형이 아닌 데이터셋
                    [2, 4, np.nan, 8, np.nan,4],
                    [2, 4, 6, 8, 10,12],
                    [np.nan, 4, np.nan, 8, np.nan,8],
                    ]).T
#print(data)
data.columns= ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# 5   8.0  4.0  12.0  8.0

from xgboost import XGBRFRegressor 
xgb =XGBRFRegressor(
    max_depth = 5,
    learning_rate=0.1,
    random_state =333,
)
imputer = IterativeImputer(
    estimator=xgb,
    max_iter=10,
    # tol=0.001, #디폴트( 얼리스탑의 기준)
    random_state=333,
)

data2 = imputer.fit_transform(data)
print(data2) 
#ex) 예를들어 x1의 결측치를 y두고 x2,x3,x4를 훈련시켜 x1을예측
#총 3번 훈련시킴
# [[ 2.          2.          2.          4.00005007]
#  [ 5.8400178   4.          4.          4.        ]
#  [ 6.          3.99998879  6.          4.00005007]
#  [ 8.          8.          8.          8.        ]
#  [10.          7.05998564 10.          7.88000011]
#  [ 8.          4.         12.          8.        ]]

print('실제 반복 횟수:', imputer.n_iter_)

# 실제 반복 횟수: 3  실제 얼리스탑핑 기능이  들어가있음