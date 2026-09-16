# [실습] Scaler 4종 비교 - 유방암 (이진 분류)
#
# keras27과 달라진 점: 스케일러를 x_train에만 fit한다.
# MinMax / Standard / MaxAbs / Robust 중 하나만 주석을 풀어서 쓰고,
# 결과가 어떻게 달라지는지 아래 기록과 비교해본다.
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer #유방암 관련 데이터셋 불러오기
from sklearn.metrics import r2_score,mean_squared_error

#1. 데이터

datasets = load_breast_cancer()
# print(datasets.DESCR)  #DESCR 실무에서는 쓸일 잘없음,사이킷런 제공데이터라
# print(datasets.feature_names)
# # ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
# #  'mean smoothness' 'mean compactness' 'mean concavity'
# #  'mean concave points' 'mean symmetry' 'mean fractal dimension'
# #  'radius error' 'texture error' 'perimeter error' 'area error'
# #  'smoothness error' 'compactness error' 'concavity error'
# #  'concave points error' 'symmetry error' 'fractal dimension error'
# #  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
# #  'worst smoothness' 'worst compactness' 'worst concavity'
# #  'worst concave points' 'worst symmetry' 'worst fractal dimension']

x = datasets.data #(569, 30)
# x = datasets['data'] #(569, 30)
y = datasets.target #(569,)

# print(x.shape,y.shape)
# print(type(x)) #'numpy.ndarray'  #판다스 자체도 넘파이로 이루어져있음
# # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# #  1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
# #  0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
# #  1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
# #  1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# #  1 1 1 1 1 1 1 0 0 0 0 0 0 1]

# print(y)  #y데이터속 라벨이 다른게 들어있을수도있어서 확인해야함
# # 0과 1의 갯수가 몇개인지 찾아보기. -numpy
# print(np.unique(y))  #[0 1]
# print(np.unique(y,return_counts=True))  # return_counts=True =(array([0, 1]), array([212, 357])) (0,1의 갯수를 파악하기위해)
# # 0과 1의 갯수가 몇개인지 찾아보기. -pandas
# print(pd.DataFrame(y).value_counts())  #print(np.unique(y,return_counts=True))랑 똑같음
# # 1    357 
# # 0    212
# print(pd.Series(y).value_counts())
# # 1    357
# # 0    212
# # Name: count, dtype: int64

x_train,x_test,y_train,y_test =train_test_split(
    x,y,
    random_state=333,
    train_size=0.8,
    stratify=y,  #x,y데이터를 나눌떄 stratify=y이걸안넣으면 x,y서로 데이터 크기가 달랐을떄 비율편차가 생길수있음
)
from sklearn.preprocessing import MinMaxScaler,StandardScaler ,MaxAbsScaler
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

# print(np.unique(y_train,return_counts=True))
# # (array([0, 1]), array([175, 280])) #>>>startify 적용후 (array([0, 1]), array([170, 285]))
# print(np.unique(y_test,return_counts=True))
# # (array([0, 1]), array([37, 77]))  #>>>startify 적용후 (array([0, 1]), array([42, 72]))

# print(x_train.shape,x_test.shape)  #(455, 30) (114, 30)
# print(y_train.shape,y_test.shape)  #(455,) (114,)



#2모델구성
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(32, activation='relu'))  #기본 디폴트값은 리니어 
model.add(Dense(1, activation='sigmoid'))  #마지막은 무조건 시그모이드 고정  


#3. 컴파일, 훈련
model.compile(loss ='binary_crossentropy',
                optimizer= 'adam',
                metrics=['acc'],        
            )  #이진분류에서는 loss = 'binary_crossentropy' 고정
# es = EarlyStopping(
#             monitor='val_loss',
#             mode= 'auto',
#             patience=20,
#             restore_best_weights=True,
#             )
start_time = time.time()  #현재 시간을 반환 ,시작시간
model.fit(x_train,y_train ,
           epochs= 100 , 
           batch_size=32,
           validation_split=0.2,
        #    callbacks =[es], 
           )
end_time = time.time()  #현재 시간을 반환 ,시작시간

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)  #시그모이드 함수를 거쳐 0,1사이 값을 반환후 >>metrics=['acc']로 후처리하면 0 OR 1로 반올림내림해서 퍼센테이지로 변환
y_pred = np.round(y_pred)# y_pred한 값이 0.11121515,0.125148이런식으로 나와서 라운드처리후  [1.] 이런식으로 변환한다음에 acc값 비교 이거 안하면 에러남

print('==============================')
print("loss:", loss[0])
print('acc:',round(loss[1],4)) #loss: 0번[0.12450382113456726,###LOSS값 (1번) 0.9473684430122375]###ACC값
print('==============================')

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,y_pred,normalize=True)
print('acc_score:', acc_score)  #acc_score: 0.9298245614035088
print('걸린시간: ',round(end_time-start_time,2),'초')

'''  
epohcs 100
cpu-걸린시간
loss: 0.22966252267360687
acc: 0.9737
acc_score: 0.9736842105263158
걸린시간:  7.65 초
gpu-걸린시간 
==============================
loss: 0.3895277678966522
acc: 0.9561
==============================
acc_score: 0.956140350877193
걸린시간:  5.42 초

'''