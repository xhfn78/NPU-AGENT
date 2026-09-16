# [실습] 함수형 모델 - 캘리포니아 주택 가격 (회귀)
#
# 같은 모델을 순차형(Sequential)과 함수형(Model) 두 가지로 각각 만들어보고
# summary()로 구조가 동일한지 확인한다.
# 함수형 문법 설명은 keras34_hamsu00.py 참고.
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout

#1. 데이터
datasets = fetch_california_housing()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.75, random_state=333,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# # 2-1. 순차형 모델 (지금까지 쓰던 방식)
# model = Sequential()
# model.add(Dense(9, input_shape=(8,), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(9, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(12, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(9, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1))
# model.summary()

# 2-2. 함수형 모델
# 함수형은 층을 변수에 담고 괄호로 이어 붙인다.
#   dense1 = Dense(30)(input1)   ← input1을 이 층에 통과시킨다는 뜻
# 위의 순차형과 층 구성이 완전히 같으므로 summary 결과도 같다. 적는 방식만 다르다.
input1 = Input(shape=(8,))
dense1 = Dense(9, name='ys1', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(9, name='ys2', activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(12,name='ys3', activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(9, name='ys4',activation='relu')(drop3)
dense5 = Dense(5, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()


#3. 컴파일, 훈련
model2.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)
from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath='./_save/keras30/keras34_hamsu01_california.keras',
    verbose=1,
)
model2.fit(x_train, y_train, epochs=300, batch_size=64, validation_split=0.2, verbose=1, callbacks=[es, mcp])

#4. 평가, 예측
loss = model2.evaluate(x_test, y_test)
y_predict = model2.predict(x_test)
print('loss:', loss)
print('r2:', r2_score(y_test, y_predict))
print('mse:', mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_predict)))

'''Dropout 적용 전/후 결과 비교
Dropout 적용 전(RobustScaler):
r2결과값: 0.7259173310645768
mse: 0.350731508708081
RMSE: 0.592225893311058

Dropout 적용 후:
실행 결과의 loss, r2, mse, RMSE 값을 아래에 기록
'''


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
# 33. Dropout 추가
# 파일: keras33_dropout01_california.py
# 이 파일에 남아 있는 수치: r2 = 0.7259173310645768
# mse = 0.350731508708081 / RMSE = 0.592225893311058
# 이 파일에는 적용 전후 표시가 없으므로 Dropout 적용 후 결과인지는 확인 필요
#
# 34. 함수형 모델로 변경
# 파일: keras34_hamsu01_california.py
# 함수형 모델로 실행한 결과: 미기록
# 기존 주석은 33번과 같은 수치를 Dropout 적용 전이라고 표기하여 전후 구분 확인 필요
# 앞선 33번 수치를 이 단계의 새 결과로 사용하지 않음
#
# ------------------------------------------------------------
# 이번 파일을 다시 실행한 결과 기록
# 실행 날짜: 
# 추가 / 변경한 내용: 
# 현재 코드 설정 (과거 결과의 실행 조건을 뜻하지 않음):
# random_state = 333 / train_size = 0.75
# epochs = 300 / batch_size = 64 / validation_split = 0.2
# 실제 훈련한 epoch 수: 
# loss: 
# r2: 
# mse: 
# RMSE: 
# 이전 비교 대상 파일: 
# 이전 결과보다 좋아진 점 / 나빠진 점: 
# 다음 실험에서 바꿀 내용: 
# ============================================================
