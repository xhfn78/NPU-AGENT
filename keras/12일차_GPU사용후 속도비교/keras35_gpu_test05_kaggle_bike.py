# [실습] Scaler 4종 비교 - 캐글 자전거 대여량 (회귀)
#
# keras27과 달라진 점: 스케일러를 x_train에만 fit한다.
# MinMax / Standard / MaxAbs / Robust 중 하나만 주석을 풀어서 쓰고,
# 결과가 어떻게 달라지는지 아래 기록과 비교해본다.
# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import time

#1. 데이터
path = './_data/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)
submission = pd.read_csv(path +'sampleSubmission.csv',index_col=0)
# print(submission)

# print(train_csv.shape) #(10886, 11)
# print(test_csv.shape) #(6493, 8)
# print(submission.shape) #(6493, 1)

# print(train_csv.info())
# print(test_csv.info())

# print(train_csv.describe())
# #######################결측치 확인 #################################
# print(train_csv.isna().sum())
# print(test_csv.isnull().sum())

# season        0            
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0
# dtype: int64


# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# dtype: int64

################# x,y 분리 ##########################
x = train_csv.drop(['casual','registered', 'count'], axis=1)
# print(x) #[10886 rows x 8 columns]

y = train_csv['count']
# print(y, y.shape)  ##(10886,)

x_train,x_test, y_train, y_test = train_test_split(x,y,
                 train_size=0.8,
                 random_state=999,

                 )
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler 
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
test_csv = scaler.transform(test_csv)



#2. 모델구성

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=8))
model.add(Dense(20,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='relu'))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam' )
start_time = time.time()
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=20,
#     restore_best_weights=True,
# )
hist = model.fit(x_train,y_train, 
                 epochs = 100,
                  batch_size=200, 
                 validation_split=0.33, 
                #  callbacks=[es]
                 )
end_time = time.time()
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
print('걸린시간 :',round(end_time - start_time,2),'초')
y_submit = model.predict(test_csv)  #test_csv를 pred 에(예측값에 넣고) y_서브밋에 저장
submission['count'] = y_submit # Y_서브밋에 저장된 내용을 서브미션 파일에 "count" 컬럼에 내용 추가 

submission.to_csv(path + 'submit/' + 'submit_0904_3.csv')


# import matplotlib.pyplot as plt
# import platform
# plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')  # 맥은 AppleGothic
# plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
# plt.legend(loc='upper right') #우측상단에 라벨표시

# plt.title('캐글 바이크 Loss') #제목
# plt.xlabel('epoch') 
# plt.ylabel('loss')
# plt.grid()  #격자표시 추가
# plt.show()
'''  
epohcs 100
cpu-걸린시간
r2 :  0.2963651418685913
mse :  22295.7421875
RMSE :  149.31758833941834
걸린시간 : 9.25 초
gpu-걸린시간 
r2 :  0.2989363670349121
mse :  22214.26953125
RMSE :  149.04452197665637
걸린시간 : 8.24 초

'''