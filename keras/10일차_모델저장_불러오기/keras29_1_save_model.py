# [실습] 모델 저장하기 1 - 훈련 "전"에 저장
#
# model.save(경로) 는 모델의 구조 + 가중치를 파일 하나로 저장한다. (.keras 확장자)
# 여기서는 훈련하기 전에 저장하므로, 저장되는 가중치는 아직 랜덤 초기값이다.
# 즉 이 파일이 저장하는 건 사실상 "모델 설계도"에 가깝다.
#28-1 카피

# import ssl
# ssl._create_default_https_context = ssl.create_default_context 다운로드 안될떄 사용할것
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.75,
    random_state=333
)


from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler  #preprocessing(전처리)
from sklearn.preprocessing import RobustScaler
##############################################################################
# scaler = MinMaxScaler()
##############################################################################

##############################################################################
# scaler = StandardScaler()
##############################################################################

##############################################################################
# scaler = MaxAbsScaler()
##############################################################################

##############################################################################
scaler = RobustScaler()
##############################################################################
# 이상치에 강력함

##############################################################################
x_train = scaler.fit_transform(x_train)
##############################################################################
x_test = scaler.transform(x_test) 

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=8,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))


model.summary()

path = './_save/keras29/'  
model.save(path + 'keras29_1_save_model.keras') #모델 구조 + 가중치 통째로 저장



exit()   # 여기서 프로그램을 끝낸다. 훈련 전 저장이 목적이라 아래 훈련 코드는 실행하지 않는다.
         # 훈련까지 해보려면 이 줄을 주석 처리하면 된다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
start_time = time.time()  #현재 시간을 반환 ,시작시간
hist = model.fit(x_train,y_train, epochs=300, batch_size=64  ,validation_split=0.2)
end_time = time.time()  #훈련 끝난 시간을 반환 , 끝시간

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2 : ' ,r2)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)


# StandardScaler
# loss: 0.5181767344474792
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 594us/step
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 438us/step
# r2결과값:  0.5950657076828227
# mse :  0.5181765626541853
# RMSE :  0.719844818453384

# maxabsscaler
# loss: 0.5213064551353455
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 571us/step
# r2결과값:  0.5926199174541247
# mse :  0.521306332589025
# RMSE :  0.7220154656162325

#RobustScaler
# loss: 0.5173594355583191
# 162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 590us/step
# r2결과값:  0.5957042535953567
# mse :  0.5173594435997324
# RMSE :  0.7192770284109818
# PS C:\study> 

# ============================================================
# California 단계별 결과 누적 비교
# 기존 주석에서 확인한 과거 기록이며, 이번에 새로 훈련한 결과는 아님.
# 단계마다 random_state, 층 구성, epochs, batch_size 등이 달라 기능 하나의 효과로 단정하지 않기.
# loss / MSE / RMSE는 낮을수록, R2는 높을수록 좋음.
# 미기록 칸은 실제 실행 후 채우기. 아래쪽 파일일수록 앞 단계 기록을 누적함.
# ============================================================
#
# 11. 기본 회귀 모델
# 파일: keras11_1_california.py
# 기존 주석 기록: loss = 0.6195971369743347
# r2 / mse / RMSE: 당시 별도 기록 없음
#
# 12. R2 / MSE / RMSE 추가
# 파일: keras12_R2_RMSE_02_california.py
# 이 단계 결과: 미기록
#
# 17. validation_split 추가
# 파일: keras17_val1_california.py
# 이 단계 결과: 미기록
# 기존 loss 주석이 11번과 같아서 별도 훈련 결과인지 확인 필요
#
# 19. loss / val_loss 그래프로 과적합 확인
# 파일: keras19_overfit1_california.py
# 이 단계 결과: 미기록
# 기존 loss 주석이 11번과 같아서 별도 훈련 결과인지 확인 필요
#
# 20. EarlyStopping 추가
# 파일: keras20_EarlyStopping1_california.py
# 이 단계 결과: 미기록
# 기존 loss 주석이 11번과 같아서 별도 훈련 결과인지 확인 필요
#
# 27. MinMaxScaler 추가
# 파일: keras27_Scaler01_california.py
# 기존 기록: loss = 0.5128337740898132
# r2 = 0.5992407312496584 / mse = 0.5128339690484836 / RMSE = 0.716124269277674
# EarlyStopping을 보완하기 전 기록이므로 현재 코드로 다시 실행한 결과는 아래에 기록
#
# 28. Scaler 종류별 비교
# 파일: keras28_Scaler01_california.py
# 기존 주석에 Scaler 이름이 적힌 기록만 해당 Scaler 결과로 정리
# MinMaxScaler: loss = 0.5144999027252197
# r2 = 0.5979388103787504 / mse = 0.514499979792781 / RMSE = 0.7172865395312956
# StandardScaler: loss = 0.5181767344474792
# r2 = 0.5950657076828227 / mse = 0.5181765626541853 / RMSE = 0.719844818453384
# MaxAbsScaler: loss = 0.5213064551353455
# r2 = 0.5926199174541247 / mse = 0.521306332589025 / RMSE = 0.7220154656162325
# RobustScaler: loss = 0.5173594355583191
# r2 = 0.5957042535953567 / mse = 0.5173594435997324 / RMSE = 0.7192770284109818
# 조건 이름이 없는 별도 기록: loss = 0.5129361748695374
# r2 = 0.5991610042912682 / mse = 0.5129359921224323 / RMSE = 0.7161954985354434
# 조건 이름이 없는 별도 기록: r2 = 0.7631599338241124
# mse = 0.3030737917688708 / RMSE = 0.5505213817544881
# EarlyStopping 보완 후 결과는 아직 미기록
#
# 29-1. 훈련 전 모델 저장
# 파일: keras29_1_save_model.py
# model.save() 후 exit()하므로 현재 파일에서는 훈련하지 않음
# 기존 Scaler 수치는 28번과 같은 과거 기록이며 이 단계의 새 결과가 아님
#
# ------------------------------------------------------------
# 이번 파일을 다시 실행한 결과 기록
# 실행 날짜: 
# 추가 / 변경한 내용: 
# 현재 코드 설정 (과거 결과의 실행 조건을 뜻하지 않음):
# random_state = 333 / train_size = 0.75
# 현재는 저장 후 exit()하여 훈련 / 평가 결과 없음.
# loss: 
# r2: 
# mse: 
# RMSE: 
# 이전 비교 대상 파일: 
# 이전 결과보다 좋아진 점 / 나빠진 점: 
# 다음 실험에서 바꿀 내용: 
# ============================================================
