# [실습] ModelCheckpoint 불러오기 - 보스턴 주택 가격 (회귀)
#
# keras31_MCP_save_03 가 저장해둔 체크포인트를 불러와서 평가만 한다.
# 먼저 keras31_MCP_save_03 를 실행해서 저장 파일을 만들어야 동작한다.

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

path = './_save/keras30/' 

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)


from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler  #preprocessing(전처리)
from sklearn.preprocessing import RobustScaler


##############################################################################
scaler = RobustScaler()
##############################################################################
scaler.fit(x_train) # x 값을  MinMaxScaler으로 실행시킬 준비
x_train = scaler.fit_transform(x_train) # 0~1 값 변환 사이로변환
x_test = scaler.transform(x_test) 

model = load_model(path +'k30_0914_1439-0088-23.6130.keras' )

#4. 평가, 예측
print("=========================================")

#4. 평가, 예측

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test ,y_predict)
print('r2: ',r2)

mse = mean_squared_error(y_test,y_predict)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 


#save
# r2:  0.7140717059140339
# RMSE :  4.878705935021808


#load
# r2:  0.7140717059140339
# RMSE :  4.878705935021808

