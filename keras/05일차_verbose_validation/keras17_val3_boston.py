#11_3 COPY
# [실습] 보스턴 주택 가격 데이터셋 - validation 적용하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam') #mse= 원값에서 예측값 뺴고 나온값을 제곱 > 다 더해서 갯수만큼 엔빵
model.fit(x_train,y_train, epochs=300, batch_size=16 ,validation_split=0.33)
          # validation_split=0.33 은 x_train의 33%를 검증용으로 떼어간다는 뜻이다.
          # 그만큼 실제 훈련에 쓰는 데이터는 줄어들지만,
          # 훈련이 진행되는 동안 val_loss로 과적합 여부를 지켜볼 수 있게 된다.


#4. 평가, 예측
print("=========================================")

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test ,y_predict)
print('r2: ',r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 

# loss: 53.61935043334961
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
# r2:  0.3558761759090454

# loss: 26.69266700744629    로스값이 대략 25 나오는데
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
# r2:  0.6793436562860996
# RMSE :  5.166494759799227  #25가 나온값을 루트 씌워서 제곱 이전으로 돌리면 5