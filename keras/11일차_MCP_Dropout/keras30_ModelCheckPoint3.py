# [실습] ModelCheckpoint 3 - 파일명에 날짜와 성능 넣기
#
# 파일명이 고정이면 실행할 때마다 이전 결과가 덮어써져서 비교를 못 한다.
# 실행 시각 + epoch + val_loss 를 파일명에 넣어 매번 다른 이름으로 남긴다.
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

#2. 모델구성
model = Sequential()
model.add(Dense(9, input_dim=8,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
##########################mcp 세이브 파일명 만들기########################
import datetime
date = datetime.datetime.now() 
print(date) #2026-09-14 11:41:17.149590
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')
print(date)
print(type(date))
path = './_save/keras30/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([path,'k30_',date,'-',filename])



model.compile(loss='mse', optimizer= 'adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience= 20,
    verbose=1,
    restore_best_weights=True,
)
# ModelCheckpoint(MCP)란?
#   훈련 도중 val_loss가 가장 좋았던 순간의 모델을 파일로 자동 저장해주는 콜백이다.
#
#   EarlyStopping의 restore_best_weights=True 와 뭐가 다른가?
#     EarlyStopping : 최적 가중치를 "메모리 안의 model"에 되돌려준다. 프로그램이 끝나면 사라진다.
#     ModelCheckpoint: 최적 시점의 모델을 "파일"로 남긴다. 나중에 다시 불러 쓸 수 있다.
#
#   주요 옵션
#     monitor='val_loss'    → 무엇을 기준으로 좋고 나쁨을 볼지
#     save_best_only=True   → 좋아졌을 때만 덮어쓴다 (False면 매 epoch 저장해서 파일이 쏟아진다)
#     filepath              → 저장할 경로와 파일명
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode= 'auto',
    save_best_only=True,
    filepath = filepath,
    verbose=1,
)




# exit()

start_time = time.time()  #현재 시간을 반환 ,시작시간
hist = model.fit(x_train,y_train, 
                 epochs=500, 
                 batch_size=32,
                 validation_split=0.2, 
                 callbacks =[es,mcp] ,
                 verbose=1,
                 )
end_time = time.time()  #훈련 끝난 시간을 반환 , 끝시간



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
# 29-2. 훈련 전 모델을 불러와 훈련
# 파일: keras29_2_load_model.py
# 이 단계 결과: 미기록 (기존 Scaler 수치는 28번과 동일)
# EarlyStopping 보완 후 결과도 새로 기록 필요
#
# 29-3. 훈련한 모델 저장
# 파일: keras29_3_save_model2.py
# 이 단계 결과: 미기록 (기존 Scaler 수치는 28번과 동일)
# EarlyStopping 보완 후 결과도 새로 기록 필요
#
# 29-4. 훈련한 모델을 불러와 평가
# 파일: keras29_4_load_model2.py
# 이 단계 결과: 미기록 (기존 Scaler 수치는 28번과 동일)
# 29-3 저장 전 결과와 비교. 다시 훈련하지 않음
#
# 29-5. 훈련 전 / 후 가중치 저장
# 파일: keras29_5_save_weights.py
# 이 단계 결과: 미기록 (기존 Scaler 수치는 28번과 동일)
# EarlyStopping 보완 후 결과도 새로 기록 필요
#
# 29-6. 훈련한 가중치를 불러와 평가
# 파일: keras29_6_load_weights.py
# 이 단계 결과: 미기록 (기존 Scaler 수치는 28번과 동일)
# 29-5 저장 전 결과와 비교. 다시 훈련하지 않음
#
# 30-1. ModelCheckpoint로 최적 모델 저장
# 파일: keras30_ModelCheckPoint1.py
# 이 파일 자체의 새 결과: 미기록 (기존 Scaler 수치는 28번과 동일)
#
# 30-2. 체크포인트를 불러와 평가
# 파일: keras30_ModelCheckPoint2_load.py
# 이 파일의 기존 주석에서 저장 / 복원 결과를 함께 기록한 수치
# 저장한 값: r2 = 0.7631599338241124 / mse = 0.3030737917688708 / RMSE = 0.5505213817544881
# 불러온 값: r2 = 0.7631599338241124 / mse = 0.3030737917688708 / RMSE = 0.5505213817544881
# 당시 기록에서는 저장 / 복원 결과가 같음
#
# 30-3. 날짜 / epoch / val_loss를 파일명에 추가
# 파일: keras30_ModelCheckPoint3.py
# 이 단계 결과: 미기록 (기존 Scaler 수치는 28번과 동일)
#
# ------------------------------------------------------------
# 이번 파일을 다시 실행한 결과 기록
# 실행 날짜: 
# 추가 / 변경한 내용: 
# 현재 코드 설정 (과거 결과의 실행 조건을 뜻하지 않음):
# random_state = 333 / train_size = 0.75
# epochs = 500 / batch_size = 32 / validation_split = 0.2
# 실제 훈련한 epoch 수: 
# loss: 
# r2: 
# mse: 
# RMSE: 
# 이전 비교 대상 파일: 
# 이전 결과보다 좋아진 점 / 나빠진 점: 
# 다음 실험에서 바꿀 내용: 
# ============================================================
