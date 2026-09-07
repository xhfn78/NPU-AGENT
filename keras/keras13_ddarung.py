#https://dacon.io/competitions/open/235576/codeshare 대회 주소
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터

path = "./_data/ddarung/"

train_csv = pd.read_csv(path + "train.csv",index_col=0 )#index_col 데이터 첫번째 ID는 Y값 추청에 전혀 영향이 없으니 데이터로 사용하지않게함
# print(train_csv)
                          #왜냐하면 훈련할때 숫자가 들어가야하는데 컬럼명은 자연어라 숫자 데이터만 남기기 위해    
#         id  hour  ...  hour_bef_pm2.5  count  <<<<컬럼명은 판다스에서 자동으로 데이터가 아닌걸로 처리함 
# 0        3    20  ...            33.0   49.0
# 1        6    13  ...            40.0  159.0
# 2        7     6  ...            19.0   26.0
# 3        8    23  ...            64.0   57.0
# 4        9    18  ...            11.0  431.0
# ...    ...   ...  ...             ...    ...
# 1454  2174     4  ...            27.0   21.0
# 1455  2175     3  ...            19.0   20.0
# 1456  2176     5  ...            21.0   22.0
# 1457  2178    21  ...            36.0  216.0
# 1458  2179    17  ...            17.0  170.0

# [1459 rows x 11 columns]  index_col=0 적용전

#       hour  hour_bef_temperature  ...  hour_bef_pm2.5  count
# id                                ...                       
# 3       20                  16.3  ...            33.0   49.0
# 6       13                  20.1  ...            40.0  159.0
# 7        6                  13.9  ...            19.0   26.0
# 8       23                   8.1  ...            64.0   57.0
# 9       18                  29.5  ...            11.0  431.0
# ...    ...                   ...  ...             ...    ...
# 2174     4                  16.8  ...            27.0   21.0
# 2175     3                  10.8  ...            19.0   20.0
# 2176     5                  18.3  ...            21.0   22.0
# 2178    21                  20.7  ...            36.0  216.0
# 2179    17                  21.1  ...            17.0  170.0

# [1459 rows x 10 columns]  index_col=0 적용후 컬럼 11 >>> 10으로 바뀜

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv)

#       hour  hour_bef_temperature  ...  hour_bef_pm10  hour_bef_pm2.5
# id                                ...                               
# 0        7                  20.7  ...           44.0            27.0
# 1       17                  30.0  ...           49.0            36.0
# 2       13                  19.0  ...           36.0            28.0
# 4        6                  22.5  ...           52.0            38.0
# 5       22                  14.6  ...           18.0            15.0
# ...    ...                   ...  ...            ...             ...
# 2148     1                  24.6  ...            NaN             NaN
# 2149     1                  18.1  ...            NaN             NaN
# 2165     9                  23.3  ...           17.0            15.0
# 2166    16                  27.0  ...           40.0            26.0
# 2177     8                  22.3  ...           30.0            24.0

# [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission)

# id         
# 0       NaN
# 1       NaN
# 2       NaN
# 4       NaN
# 5       NaN
# ...     ...
# 2148    NaN
# 2149    NaN
# 2165    NaN
# 2166    NaN
# 2177    NaN

# [715 rows x 1 columns]

# print(train_csv.shape) #(1459, 10)
# print(test_csv.shape) #(715, 9)
# print(submission.shape) #(715, 1)

# print(train_csv.info())

# <class 'pandas.DataFrame'>
# Index: 1459 entries, 3 to 2179
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    1459 non-null   int64  
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64  
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64    결측치 보는법 1457~1342 데이터가 많은곳과 빠진곳이있음
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)

# exit()

##############################결측치 처리 1.삭제###################################

train_csv = train_csv.dropna() # 결측치(NaN) 있는 ROW 행 삭제후 다시 train.csv에 넣어줌 
# print(train_csv)   #[1328 rows x 10 columns]

############################train_cs를 x와 y로 분리##################################

x = train_csv.drop(['count'], axis=1)  #열(컬럼) 삭제  drop(['컬럼명 넣으면됨'])

# print(x)


y = train_csv['count']
# print(y)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=100
)



#2.모델구성

model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3.컴파일 ,훈련

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y , epochs= 100 , batch_size=40)


#4.평가 ,예측

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('결과값: ' ,r2)

mse = mean_squared_error(y_test,y_predict)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 


# loss: 2726.558837890625
# 11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# 결과값:  0.6003555849928867
# RMSE :  52.216461878806555