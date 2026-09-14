# [실습] Scaler 4종 비교 - 당뇨병 (회귀)
#
# keras27과 달라진 점: 스케일러를 x_train에만 fit한다.
# MinMax / Standard / MaxAbs / Robust 중 하나만 주석을 풀어서 쓰고,
# 결과가 어떻게 달라지는지 아래 기록과 비교해본다.
from sklearn.datasets import fetch_california_housing, load_diabetes #캘리포니아 집값 데이터셋,로드 디아벳
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape,y.shape) #(442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    random_state=221,
    train_size=0.75, 

)
from sklearn.preprocessing import MinMaxScaler,StandardScaler ,MaxAbsScaler #preprocessing(전처리)
from sklearn.preprocessing import RobustScaler
##############################################################################
# scaler = MinMaxScaler()
##############################################################################
# MinMaxScaler
#   X_scaled = (X - X_min) / (X_max - X_min)
#   fit한 데이터의 최솟값을 0, 최댓값을 1로 만든다.
#   단점: 이상치가 하나라도 있으면 Min/Max 자체가 그 이상치로 잡혀서 크게 흔들린다.



##############################################################################
# scaler = StandardScaler()
##############################################################################
# StandardScaler
#   z = (x - 평균) / 표준편차
#   평균을 0, 표준편차를 1로 만든다.
#   z = 1 은 "평균보다 표준편차 1개만큼 위"라는 뜻이고, z = -2 는 "평균보다 2개만큼 아래"라는 뜻이다.
#   단점: 평균과 표준편차도 이상치의 영향을 받는다.



##############################################################################
# scaler = MaxAbsScaler()
##############################################################################
# MaxAbsScaler
#   X_scaled = X / max(|X|)
#   그 feature의 최대 절댓값으로 나눈다. 예) [-50, 0, 100] → [-0.5, 0, 1.0]
#   주의: 최솟값이 항상 -1이 되는 게 아니라, 절댓값이 가장 큰 값만 ±1이 된다.
#   단점: 최대 절댓값을 기준으로 삼기 때문에 큰 이상치에 민감하다.


##############################################################################
scaler = RobustScaler()
##############################################################################
# 이상치에 강력함
# RobustScaler
#   X_scaled = (X - 중앙값) / IQR       (IQR = 3사분위수 - 1사분위수)
#   중심을 평균 대신 중앙값으로, 폭을 표준편차 대신 IQR로 잡는다.
#   중앙값과 IQR은 이상치 하나에 잘 흔들리지 않아서 이상치에 강하다.
#   단, 이상치를 제거하는 게 아니라 이상치 때문에 스케일링 기준이 왜곡되는 걸 줄이는 것이다.

# scaler.fit(x_train)   # fit만 하는 줄. 아래에서 fit_transform으로 한 번에 하므로 중복이라 꺼둔다.
# [ 스케일러는 x_train에만 fit한다 ]
#   x_test와 실전 데이터는 x_train에서 학습한 기준으로 transform만 해야 한다.
#   test 데이터의 정보가 스케일러에 미리 반영되면 평가를 믿을 수 없게 되기 때문이다.
x_train = scaler.fit_transform(x_train)   # x_train 기준을 학습(fit)하고 동시에 변환(transform)
x_test = scaler.transform(x_test)         # test는 transform만 (fit 금지)

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode='auto',
    patience=15,
    restore_best_weights=True
)
hist = model.fit(x_train,y_train, 
                 epochs=3000, 
                 batch_size=10 ,
                 validation_split=0.2,
                 callbacks =[es]
                 )


#4. 평가, 예측
print("=========================================")
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
# results = model.predict(x)
# print('결과값: ' ,results)
#랜덤 442
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2857.6667 
# loss: 2857.666748046875
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 결과값:  [[210.36603 ]

#랜덤 221
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2459.2292 
# loss: 2459.229248046875
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 

#랜덤 3333
# 42/42 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2882.3848 
# =========================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 3071.2979 
# loss: 3071.2978515625
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 결과값:  [[204.20064 ]
# import matplotlib.pyplot as plt
# import platform
# plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')  # 맥은 AppleGothic
# plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
# plt.legend(loc='upper right') #우측상단에 라벨표시

# plt.title('당뇨병 Loss') #제목
# plt.xlabel('epoch') 
# plt.ylabel('loss')
# plt.grid()  #격자표시 추가
# plt.show()


'''
하이퍼 파라미터 튜닝
#1. 데이터부분
random_state
train_size
#2.
레이어의 깊이
노드의갯수
#3
epoch
batch_size
'''


"""
1차시도
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2637.873291015625
r2결과값:  0.5483979249211794
mse :  2637.8732195134085
RMSE :  51.36022994023107
"""

"""
2차시도 --- MINMAX-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2574.98388671875
r2결과값:  0.5591645248939214
mse :  2574.983947518029
RMSE :  50.744299655409854
"""



"""
3차시도 --- standard-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2451.989013671875
r2결과값:  0.5802211882185655
mse :  2451.9889230450026
RMSE :  49.5175617639338
"""

"""
3차시도 --- standard-scaler 적용
random : 221
train_size = 0.75
epochs = 30000
batch_size = 10
결과
loss: 2661.037353515625
r2결과값:  0.5444322418537058
mse :  2661.0373494810156
RMSE :  51.58524352449076
"""

