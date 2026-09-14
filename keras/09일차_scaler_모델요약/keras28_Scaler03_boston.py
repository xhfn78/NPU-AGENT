# [실습] Scaler 4종 비교 - 보스턴 주택 가격 (회귀)
#
# keras27과 달라진 점: 스케일러를 x_train에만 fit한다.
# MinMax / Standard / MaxAbs / Robust 중 하나만 주석을 풀어서 쓰고,
# 결과가 어떻게 달라지는지 아래 기록과 비교해본다.
#11_3 COPY
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error


#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)


from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler  #preprocessing(전처리)
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
model.add(Dense(3, input_dim=13))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam') #mse= 원값에서 예측값 뺴고 나온값을 제곱 > 다 더해서 갯수만큼 엔빵
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode= 'auto',
    patience= 15,
    restore_best_weights=True,
)
hist = model.fit(x_train,y_train, 
                 epochs=500, 
                 batch_size=16 ,
                 validation_split=0.2,
                 callbacks =[es],
                 )


#4. 평가, 예측
print("=========================================")

loss = model.evaluate(x_test,y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test ,y_predict)
print('r2: ',r2)

mse = mean_squared_error(y_test,y_predict)

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
# RMSE :  5.166494759799227  #25가 나온값을 루트 쓰위서 제곱 이전으로 돌리면 5
# import matplotlib.pyplot as plt
# import platform
# plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')  # 맥은 AppleGothic
# plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
# plt.legend(loc='upper right') #우측상단에 라벨표시

# plt.title('보스턴 Loss') #제목
# plt.xlabel('epoch') 
# plt.ylabel('loss')
# plt.grid()  #격자표시 추가
# plt.show()

"""
2차시도 --- MINMAX 적용
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 23.3464
loss: 23.34636878967285
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
r2:  0.7195424271033053
RMSE :  4.8318079136533605

random : 221
train_size = 0.75
epochs = 500
batch_size = 16
결과
loss: 2574.98388671875
r2결과값:  0.5591645248939214
mse :  2574.983947518029
RMSE :  50.744299655409854
"""
#StandardScaler
"""
3차시도 --- standard 적용
loss: 24.623424530029297
r2:  0.7042012453817736
RMSE :  4.962199786564942

"""

#maxabsscaler
'''
4차시도 --- standard 적용
loss: 24.623424530029297
r2:  0.7115445227425319
RMSE :  4.900218776088198
'''