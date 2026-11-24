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

# K-Nearest Neighbors
# 거리 계산: 결측치가 없는 특성(feature)들을 
# 기준으로 유클리드 거리(nan_euclidean_distances)를 계산하여 
# 가장 가까운 K개의 이웃을 찾습니다.
imputer = KNNImputer(n_neighbors=5) #디폴트 n_neighbors=5
# imputer = KNNImputer(n_neighbors=2) #디폴트 n_neighbors=5
data2 = imputer.fit_transform(data)
print(data2)

# [[ 2.          2.          2.          6.66666667]
#  [ 6.8         4.          4.          4.        ]
#  [ 6.          4.5         6.          6.66666667]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.5        10.          6.66666667]
#  [ 8.          4.         12.          8.        ]]


