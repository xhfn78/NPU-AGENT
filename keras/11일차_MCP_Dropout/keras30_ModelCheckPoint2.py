# [실습] ModelCheckpoint 2 - 저장한 체크포인트 불러오기
#
# keras30_ModelCheckPoint1에서 저장한 파일을 load_model로 불러온다.
# 훈련이 끝난 최적 시점의 모델이므로 fit 없이 바로 평가할 수 있다.
#30-1카피

# import ssl
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler  #preprocessing(전처리)
from sklearn.preprocessing import RobustScaler
import numpy as np
import time

path = './_save/keras30/'  

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.75,
    random_state=333
)


##############################################################################
scaler = RobustScaler()
##############################################################################
# 이상치에 강력함

##############################################################################
x_train = scaler.fit_transform(x_train)
##############################################################################
x_test = scaler.transform(x_test) 




model = load_model(path + 'keras30_mcp1.keras')



#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2 : ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 



'''
저장한 값들
r2결과값:  0.7631599338241124
mse :  0.3030737917688708
RMSE :  0.5505213817544881


불러온 값들  
r2결과값:  0.7631599338241124
mse :  0.3030737917688708
RMSE :  0.5505213817544881
'''

