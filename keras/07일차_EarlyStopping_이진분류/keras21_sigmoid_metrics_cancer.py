import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer #유방암 관련 데이터셋 불러오기
from sklearn.metrics import r2_score,mean_squared_error

#1.데이터

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
# #  1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
# #  1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
# #  1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
# #  1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
# #  1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
# #  0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
# #  1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
# #  1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
# #  0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
# #  1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
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


#3.컴파일 ,훈련
model.compile(loss ='binary_crossentropy',
                optimizer= 'adam',
                metrics=['acc'],        
            )  #이진분류에서는 loss = 'binary_crossetropy' 고정
es = EarlyStopping(
            monitor='val_loss',
            mode= 'auto',
            patience=15,
            restore_best_weights=True,
            )
strat_time = time.time()  #현재 시간을 반환 ,시작시간
model.fit(x_train,y_train ,
           epochs= 500 , 
           batch_size=32,
           validation_split=0.2,
           callbacks =[es], 
           )
end_time = time.time()  #현재 시간을 반환 ,시작시간

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('==============================')
print("loss:", loss[0])
print('acc:',round(loss[1],4)) #loss: 0번[0.12450382113456726,###LOSS값 (1번) 0.9473684430122375]###ACC값
print('==============================')
y_pred = model.predict(x_test)  #시그모이드 함수를 거쳐 0,1사이 값을 반환후 >>metrics=['acc']로 후처리하면 0 OR 1로 반올림내림해서 퍼센테이지로 변환
y_pred = np.round(y_pred)# y_pred한 값이 0.11121515,0.125148이런식으로 나와서 라운드처리후  [1.] 이런식으로 변환한다음에 acc값 비교 이거 안하면 에러남
print(y_pred[:10])
# print(y_pred[:10])
from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test,y_pred,normalize=True)
print('acc_score:', acc_score)  #acc_score: 0.9298245614035088
# ==============================
# loss: 0.1361425369977951
# acc: 0.9298
# ==============================

# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict) 
# print('r2결과값: ' ,r2)

# mse = mean_squared_error(y_test,y_predict)
# print('mse : ', mse)

# def RMSE(y_test, y_predict):  #RMSE 함수정의
#     return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

# rmse = RMSE(y_test, y_predict)

# print('RMSE : ', rmse) 

# import matplotlib.pyplot as plt
# plt.rc('font', family='Malgun Gothic')  #맑은 고딕 폰트 적용 한글꺠짐 방지
# plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'] ,c='blue', label='val_loss')
# plt.legend(loc='upper right') #우측상단에 라벨표시

# plt.title('유방암 Loss') #제목
# plt.xlabel('epoch') 
# plt.ylabel('loss')
# plt.grid()  #격자표시 추가
# plt.show()
