import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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

imputer = SimpleImputer(strategy='most_frequent') #최빈값
data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.  2.  2.  8.]
#  [ 8.  4.  4.  4.]   <<<<<<<<<결측치가 최빈값으로 채워짐
#  [ 6.  4.  6.  8.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  8.]
#  [ 8.  4. 12.  8.]]

imputer2 = SimpleImputer(strategy='constant',fill_value=777,)  #특정값 넣기
data3 = imputer2.fit_transform(data)
print(data3)
# [[  2.   2.   2. 777.]  <<<<<<결측치 777로 채워짐
#  [777.   4.   4.   4.]
#  [  6. 777.   6. 777.]
#  [  8.   8.   8.   8.]
#  [ 10. 777.  10. 777.]
#  [  8.   4.  12.   8.]]



imputer3 = SimpleImputer(
#     strategy=lambda x: np.max(x) #최대값
    strategy=lambda x: np.min(x)  #최솟값 
    )  
data4 = imputer3.fit_transform(data)
print(data4)

# [[ 2.  2.  2.  8.]
#  [10.  4.  4.  4.]
#  [ 6.  8.  6.  8.]
#  [ 8.  8.  8.  8.]
#  [10.  8. 10.  8.]
#  [ 8.  4. 12.  8.]]

# [[ 2.  2.  2.  4.]
#  [ 2.  4.  4.  4.]
#  [ 6.  2.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  2. 10.  4.]
#  [ 8.  4. 12.  8.]]

