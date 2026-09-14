# [실습] EarlyStopping 적용 - 캐글 자전거 대여량
# https://www.kaggle.com/competitions/bike-sharing-demand/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


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
from tensorflow.keras.callbacks import EarlyStopping
# EarlyStopping 옵션 설명은 keras20_EarlyStopping1_california.py 에 자세히 적어뒀다.
#   monitor='val_loss'          → 검증 loss를 감시
#   mode='auto'                 → 작아야 좋은지 커야 좋은지 케라스가 알아서 판단
#   patience                    → 개선 없이 몇 epoch까지 참을지 (단위는 epoch)
#   restore_best_weights=True   → 멈춘 시점이 아니라 가장 좋았던 시점의 W, b로 되돌림
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)
hist = model.fit(x_train,y_train, epochs = 2000, batch_size=200, validation_split=0.33, callbacks=[es])

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

y_submit = model.predict(test_csv)  #test_csv를 pred 에(예측값에 넣고) y_서브밋에 저장
submission['count'] = y_submit # Y_서브밋에 저장된 내용을 서브미션 파일에 "count" 컬럼에 내용 추가 

submission.to_csv(path + 'submit/' + 'submit_0904_3.csv')


import matplotlib.pyplot as plt
import platform
# 한글 깨짐 방지. 윈도우는 맑은 고딕, 맥은 AppleGothic을 써야 한다.
plt.rc('font', family='Malgun Gothic' if platform.system()=='Windows' else 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자 나올때 깨짐방지
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][2:] ,c='blue', label='val_loss')
plt.legend(loc='upper right') #우측상단에 라벨표시

plt.title('캐글 바이크 Loss') #제목
plt.xlabel('epoch') 
plt.ylabel('loss')
plt.grid()  #격자표시 추가
plt.show()


'''
하이퍼 파라미터 튜닝
#1.데이터 부분
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
random : 333
train_size = 0.75
epochs = 200
batch_size = 1000
결과
loss =24249.419921875
rmse: 155.72223995186107
r2 :0.24
"""