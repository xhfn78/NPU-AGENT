# [실습] 캘리포니아 주택 가격 데이터셋 - validation 적용하기
# import ssl
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=42
)

print(x.shape,y.shape) #(20640, 8) (20640,)

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=8))
model.add(Dense(9))
model.add(Dense(12))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train,y_train, epochs=200, batch_size=16  ,validation_split=0.33)
          # validation_split=0.33 은 x_train의 33%를 검증용으로 떼어간다는 뜻이다.
          # 그만큼 실제 훈련에 쓰는 데이터는 줄어들지만,
          # 훈련이 진행되는 동안 val_loss로 과적합 여부를 지켜볼 수 있게 된다.


#4. 평가, 예측
print("=========================================")

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
results = model.predict(x)
print('결과값: ' ,results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)

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

# ============================================================
# California 단계별 결과 누적 비교
# 기존 주석에서 확인한 과거 기록이며, 이번에 새로 훈련한 결과는 아님.
# 단계마다 random_state, 층 구성, epochs, batch_size 등이 달라 기능 하나의 효과로 단정하지 않기.
# loss / MSE / RMSE는 낮을수록, R2는 높을수록 좋음.
# 미기록 칸은 실제 실행 후 채우기. 아래쪽 파일일수록 앞 단계 기록을 누적함.
# ============================================================
#
# 11. 기본 회귀 모델
# 파일: keras11_1_califonia.py
# 기존 주석 기록: loss = 0.6195971369743347
# r2 / mse / RMSE: 당시 별도 기록 없음
#
# 12. R2 / MSE / RMSE 추가
# 파일: keras12_R2_RMSE_02_california.py
# 이 단계 결과: 미기록
#
# 17. validation_split 추가
# 파일: keras17_val1_califonia.py
# 이 단계 결과: 미기록
# 기존 loss 주석이 11번과 같아서 별도 훈련 결과인지 확인 필요
#
# ------------------------------------------------------------
# 이번 파일을 다시 실행한 결과 기록
# 실행 날짜: 
# 추가 / 변경한 내용: 
# 현재 코드 설정 (과거 결과의 실행 조건을 뜻하지 않음):
# random_state = 42 / train_size = 미지정 (기본 분할)
# epochs = 200 / batch_size = 16 / validation_split = 0.33
# 실제 훈련한 epoch 수: 
# loss: 
# r2: 
# mse: 
# RMSE: 
# 이전 비교 대상 파일: 
# 이전 결과보다 좋아진 점 / 나빠진 점: 
# 다음 실험에서 바꿀 내용: 
# ============================================================
