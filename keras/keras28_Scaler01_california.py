#19-1 카피

# import ssl
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것


from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import time

#1.데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
'''
MinMaxScaler

계산법 :
원값 - min    
----------
max - min  
'''





x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.75,
    random_state=333
)

from sklearn.preprocessing import MinMaxScaler  #preprocessing(전처리)
scaler = MinMaxScaler()
scaler.fit(x_train) # x 값을  MinMaxScaler으로 실행시킬 준비
x_train = scaler.transform(x_train) # 0~1 값 변환 사이로변환
x_test = scaler.transform(x_test) 
print(x)
print(np.min(x_train),np.max(x_train))  #0.0-> min값    1.0000000000000002 -> max값
print(np.min(x_test),np.max(x_test))  #0.0-> min값    1.0000000000000002 -> max값


#2.모델구성
model = Sequential()
model.add(Dense(9, input_dim=8))
model.add(Dense(9))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))


#3.컴파일,훈련
model.compile(loss='mse', optimizer= 'adam')
strat_time = time.time()  #현재 시간을 반환 ,시작시간
hist = model.fit(x_train,y_train, epochs=300, batch_size=64  ,validation_split=0.2)
end_time = time.time()  #훈련 끝난 시간을 반환 , 끝시간



#4.평가 ,예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2결과값: ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 

# print('걸린시간 :',round(end_time - strat_time,2),'초')

# print('====================history=======================')
# print(hist) #<keras.src.callbacks.history.History object at 0x000001FABB96A490>
# print('====================hist.history=======================')
# print(hist.history)
# print('====================loss=======================')
# print(hist.history['loss'])
# print('====================val_loss=======================')
# print(hist.history['val_loss'])


# =========================================
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 682us/step - loss: 0.6196
# loss: 0.6195971369743347
# 645/645 ━━━━━━━━━━━━━━━━━━━━ 0s 350us/step
# 결과값:  [[4.222328 ]
#  [3.9601626]
#  [3.791273 ]
#  ...
#  [0.7560466]
#  [0.8599502]
#  [1.0874628]]


# loss: 0.5128337740898132
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 602us/step
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 416us/step
# r2결과값:  0.5992407312496584
# mse :  0.5128339690484836
# RMSE :  0.716124269277674

# loss: 0.5144999027252197        ------minmax 적용후----
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 634us/step
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 364us/step
# r2결과값:  0.5979388103787504
# mse :  0.514499979792781
# RMSE :  0.7172865395312956


# loss: 0.5129361748695374
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 737us/step
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 431us/step
# r2결과값:  0.5991610042912682
# mse :  0.5129359921224323
# RMSE :  0.7161954985354434
