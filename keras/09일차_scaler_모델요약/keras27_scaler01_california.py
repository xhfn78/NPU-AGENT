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
from sklearn.preprocessing import MinMaxScaler  #preprocessing(전처리)

scaler = MinMaxScaler()
scaler.fit(x) # x 값을  MinMaxScaler으로 실행시킬 준비
x = scaler.transform(x) # 0~1 값 변환 사이로변환 
print(x)
print(np.min(x),np.max(x))  #0.0-> min값    1.0000000000000002 -> max값





x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.75,
    random_state=333
)

print(x.shape,y.shape) #(20640, 8) (20640,)

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
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True,
)
hist = model.fit(x_train,y_train, epochs=100, batch_size=32  ,validation_split=0.2, callbacks=[es])
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

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')  #맑은 고딕 폰트 적용 한글꺠짐 방지
plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
plt.legend(loc='upper right') #우측상단에 라벨표시

plt.title('캘리포니아 Loss') #제목
plt.xlabel('epoch') 
plt.ylabel('loss')
plt.grid()  #격자표시 추가
plt.show()



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
# 파일: keras27_scaler01_california.py
# 기존 기록: loss = 0.5128337740898132
# r2 = 0.5992407312496584 / mse = 0.5128339690484836 / RMSE = 0.716124269277674
# EarlyStopping을 보완하기 전 기록이므로 현재 코드로 다시 실행한 결과는 아래에 기록
#
# ------------------------------------------------------------
# 이번 파일을 다시 실행한 결과 기록
# 실행 날짜: 
# 추가 / 변경한 내용: 
# 현재 코드 설정 (과거 결과의 실행 조건을 뜻하지 않음):
# random_state = 333 / train_size = 0.75
# epochs = 100 / batch_size = 32 / validation_split = 0.2
# 실제 훈련한 epoch 수: 
# loss: 
# r2: 
# mse: 
# RMSE: 
# 이전 비교 대상 파일: 
# 이전 결과보다 좋아진 점 / 나빠진 점: 
# 다음 실험에서 바꿀 내용: 
# ============================================================
