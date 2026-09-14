# [실습] Scaler 4종 비교 - 산탄데르 (이진 분류, 캐글)
#
# keras27과 달라진 점: 스케일러를 x_train에만 fit한다.
# MinMax / Standard / MaxAbs / Robust 중 하나만 주석을 풀어서 쓰고,
# 결과가 어떻게 달라지는지 아래 기록과 비교해본다.
###이진 븐류를 다중분류로 변형해서 테스트해보기###
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1. 데이터
path = './_data/kaggle_santander/'   #<<< 상대경로 (윈도우/맥 어디서나 동작)
# path = 'c://study//_data//kaggle_santander//'   #<<< 윈도우 절대경로. // 두 개 써도 가능하지만 맥에서는 안 됨
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0 )
# print(test_csv)
submission_csv = pd.read_csv(path +'sample_submission.csv',index_col=0)

x = train_csv.drop(['target'], axis=1)  #타겟 열 한개뺴기
# # print(x) #[10886 rows x 8 columns]

y = train_csv['target']
# print(x.shape,y.shape)  #(200000, 200) (200000,)
# print(y)
# print(np.unique(y,return_counts=True)) 
 #(array([0, 1]), array([179902,  20098]))
##################원 핫1, to_categorycal####################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(x.shape,y.shape)  #(200000, 200) (200000,)
# print(y)
# print(np.unique(y,return_counts=True)) 
#  #(array([0, 1]), array([179902,  20098]))
# # exit()



x_train,x_test,y_train,y_test =train_test_split(
    x,y,
    random_state=23,
    train_size=0.8,
    stratify=y,  #x,y데이터를 나눌떄 stratify=y이걸안넣으면 x,y서로 데이터 크기가 달랐을떄 비율편차가 생길수있음
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



# #2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=200, activation= 'relu'))
model.add(Dense(200, activation= 'relu'))
model.add(Dense(300,activation= 'relu'))
model.add(Dense(400, activation= 'relu'))
model.add(Dense(200, activation= 'relu'))
model.add(Dense(2,activation= 'softmax'))

#3. 컴파일, 훈련
model.compile(loss ='categorical_crossentropy',
                optimizer= 'adam',
                metrics=['acc'],        
            )  #이진분류에서는 loss = 'binary_crossentropy' 고정
es = EarlyStopping(
            monitor='val_loss',
            mode= 'auto',
            patience=100,
            restore_best_weights=True,
            )
start_time = time.time()  #현재 시간을 반환 ,시작시간
model.fit(x_train,y_train,
           epochs=100 , 
           batch_size=40000,
           validation_split=0.2,
           callbacks =[es], 
           )
end_time = time.time()  #현재 시간을 반환 ,시작시간

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)#이벨류에이트 할떄는 자체적으로 라운드처리해서 acc가 나옴
print("loss:", loss)
print('걸린시간 :',round(end_time - start_time,2),'초')

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test,y_predict,)
print('acc_score:', acc_score)  #acc_score: 0.9298245614035088

y_submit = model.predict(test_csv)  #test_csv를 pred 에(예측값에 넣고) y_서브밋에 저장
y_submit = np.argmax(y_submit, axis=1)

submission_csv['target'] = y_submit # Y_서브밋에 저장된 내용을 서브미션 파일에 "count" 컬럼에 내용 추가 
submission_csv.to_csv(path + 'submit/' + 'submit_0910_1.csv')



"""
1차시도
random : 666
train_size = 0.8
epochs = 1000
batch_size = 10000
결과
loss: [0.25124090909957886, 0.9091500043869019]
걸린시간 : 156.1 초
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 741us/step
acc_score: 0.90915
6250/6250 ━━━━━━━━━━━━━━━━━━━━ 4s 674us/step 
"""

"""
2차시도----스케일러 적용----
random : 666
train_size = 0.8
epochs = 1000
batch_size = 10000
결과
loss: [0.2588488459587097, 0.9073500037193298]
걸린시간 : 49.72 초
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 665us/step
acc_score: 0.90735
6250/6250 ━━━━━━━━━━━━━━━━━━━━ 4s 616us/step
"""

"""
3차시도----스케일러 적용----
random : 666
train_size = 0.8
epochs = 1000
batch_size = 10000
결과
loss: [0.2430790215730667, 0.9104750156402588]
걸린시간 : 50.11 초
1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 627us/step
acc_score: 0.910475
6250/6250 ━━━━━━━━━━━━━━━━━━━━ 4s 613us/step 
"""