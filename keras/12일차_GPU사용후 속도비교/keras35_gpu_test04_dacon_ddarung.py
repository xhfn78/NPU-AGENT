# [실습] Scaler 4종 비교 - 따릉이 대여량 (회귀, 데이콘)
#
# keras27과 달라진 점: 스케일러를 x_train에만 fit한다.
# MinMax / Standard / MaxAbs / Robust 중 하나만 주석을 풀어서 쓰고,
# 결과가 어떻게 달라지는지 아래 기록과 비교해본다.
#https://dacon.io/competitions/open/235576/codeshare 대회 주소
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import time

#1. 데이터

path = './_data/ddarung/'      #<<< 상대경로 (윈도우/맥 어디서나 동작)
# path = 'c:\study\_data\ddarung\'   #<<< 윈도우 절대경로. \ 두 개 써도 가능하지만 맥에서는 안 됨
train_csv = pd.read_csv(path + "train.csv",index_col=0 )#index_col 데이터 첫번째 ID는 Y값 추청에 전혀 영향이 없으니 데이터로 사용하지않게함
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "submission.csv", index_col=0)
train_csv = train_csv.dropna() # 결측치(NaN) 있는 ROW 행 삭제후 다시 train.csv에 넣어줌 
x = train_csv.drop(['count'], axis=1)  #열(컬럼) 삭제  drop(['컬럼명 넣으면됨'])
y = train_csv['count']
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=666,
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


#####################submit 작업 ################################
# print(test_csv.info())
# print(test_csv.shape) 

#  #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    715 non-null    int64  
#  1   hour_bef_temperature    714 non-null    float64


######################결측치 처리 2.평균값 넣기 ####################
# test_csv = test_csv.fillna(test_csv.mean())   ##
test_csv = test_csv.fillna(test_csv.mean())
test_csv = scaler.transform(test_csv)
# print(test_csv.info()) #(715, 9)
# print(test_csv.shape) #(715, 9)

#  #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    715 non-null    int64  
#  1   hour_bef_temperature    715 non-null    float64
#  2   hour_bef_precipitation  715 non-null    float64
#  3   hour_bef_windspeed      715 non-null    float64
# exit()

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=20,
#     restore_best_weights=True,
# )
hist = model.fit(x_train,y_train ,
                epochs= 100 , 
                batch_size=32,
                validation_split=0.2, 
                # callbacks=[es]
                )
end_time = time.time()
#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ' ,r2)

y2_pred = model.predict(test_csv)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 
print('걸린시간 :',round(end_time - start_time,2),'초')



######################submisson.csv 만들기 // count 컬럼에 값 넣어준다.####################
# print(submission)
#       count
# id         
# 0       NaN
# 1       NaN
# 2       NaN

y_submit = model.predict(test_csv)
submission['count'] = y_submit
# print(submission)
# print(submission.shape)

#  count
# id             
# 0     -6.693819
# 1    -41.792511
# 2     46.420265
# [715 rows x 1 columns]
# (715, 1)

submission.to_csv(path + 'submit/' + 'submit_0904_1148.csv')

# import matplotlib.pyplot as plt
# import platform
# # 한글 깨짐 방지. 윈도우는 맑은 고딕, 맥은 AppleGothic을 써야 한다.
# plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')
# plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
# plt.legend(loc='upper right') #우측상단에 라벨표시

# plt.title('따릉이 Loss') #제목
# plt.xlabel('epoch') 
# plt.ylabel('loss')
# plt.grid()  #격자표시 추가
# plt.show()


'''  epohcs 100
cpu-걸린시간
mse :  2542.264189448448
RMSE :  50.42087057408319
걸린시간 : 7.99 초

gpu-걸린시간 
mse :  2449.3428905227083
RMSE :  49.49083642981505
걸린시간 : 6.38 초

'''