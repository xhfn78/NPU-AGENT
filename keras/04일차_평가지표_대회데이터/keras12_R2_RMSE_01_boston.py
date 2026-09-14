#11_3 COPY
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# mse = 원래값에서 예측값을 뺀 값을 제곱 > 다 더해서 개수만큼 나눔(엔빵)
# 역전파가 어디를 얼마나 고칠지 계산하고, optimizer(adam)가 실제로 고친다.
model.fit(x_train, y_train, epochs=300, batch_size=16)

#4. 평가, 예측
print("=========================================")

loss = model.evaluate(x_test, y_test)
print("loss:", loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error

# loss(mse) 하나만 보면 "이 값이 좋은 건지" 알기 어렵다.
# 데이터마다 target의 크기가 달라서 mse의 단위도 달라지기 때문이다.
# 그래서 평가 지표를 두 개 더 쓴다.
#
#  R2   : 0~1 사이. 1에 가까울수록 잘 맞힌 것. 데이터 크기와 무관해서 비교하기 좋다.
#  RMSE : mse에 루트를 씌운 값. 단위가 원래 y와 같아져서 "평균 몇 정도 틀렸나"로 읽힌다.

r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

mse = mean_squared_error(y_test, y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  # np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)

# loss: 53.61935043334961
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
# r2:  0.3558761759090454

# loss: 26.69266700744629    로스값이 대략 25 나오는데
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
# r2:  0.6793436562860996
# RMSE :  5.166494759799227  #25가 나온값을 루트 씌워서 제곱 이전으로 돌리면 5
