import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan],
                    ]).T

print(data)
'''
      0    1     2    3
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''
data.columns= ['x1', 'x2', 'x3', 'x4']
print(data)
'''
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''

# 0. 결측치 확인
print(data.isnull())
'''
      x1     x2     x3     x4
0  False  False  False   True
1   True  False  False  False
2  False   True  False   True
3  False  False  False  False
4  False   True  False   True
True가 결측치
'''
# 결측치 개수 확인
print(data.isnull().sum())
'''
x1    1
x2    2
x3    0
x4    3
dtype: int64
'''
print(data.info())
''''
<class 'pandas.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   x1      4 non-null      float64
 1   x2      3 non-null      float64
 2   x3      5 non-null      float64
 3   x4      2 non-null      float64
dtypes: float64(4)
memory usage: 292.0 bytes
None
'''

# 1. 결측치 삭제
# print(data.dropna())
'''
    x1   x2   x3   x4
3  8.0  8.0  8.0  8.0
결측치가 있는 column빼고 다 사라짐
dropna의 default는 행
'''
# print(data.dropna(axis=0))  # 행 삭제
#   x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0

# print(data.dropna(axis=1))  # 열 삭제
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

# 2-1. 특정값 - 평균

means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)
#      x1        x2    x3   x4
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0

#2-2. 특정값 - 중위값

# med = data.median()
# print(med)
# data3 = data.fillna(med)
# print(data3)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

#2-3. 특정값  -0으로 채워버리기
# data4 =data.fillna(0)
# print(data4)
#     x1   x2    x3   x4
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0

#2-32. 특정값  -777으로 채워버리기
# data4 =data.fillna(777)
# print(data4)

#2-4.특정값 - ffill(통상 마지막 값,) (시계열에서 많이쓰임)

# data5 = data.ffill()
# print(data5)
# x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN<<< 맨앞 nan은 처리가안되기때문에 다시 처리해줘야함
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

#2.5 .특정값 - bfill(통상 첫번째 ),(시계열에서 많이씀)
# data6 = data.bfill()
# print(data6)
#     x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN<<<가장 마지막행은 채울값이 없어서 Nan

##################특정 컬럼만 ##########################
print('===========================================')
means = data['x1'].mean()
print(means) #6.5 나옴

med = data['x4'].median()
print(med) #6.0

data['x1'] = data['x1'].fillna(means)
data['x2'] = data['x2'].ffill()
data['x4'] = data['x4'].fillna(med)
print(data)
#   x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   6.5  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  6.0

