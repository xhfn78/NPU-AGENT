# 캐글 자전거 대여량 예측
# 이 링크에서 .csv 파일을 다운로드 받아야 한다.
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


################# x,y 분리 ##########################
# count 말고 casual, registered도 같이 지우는 이유:
#   casual(비회원) + registered(회원) = count 라서
#   이 둘을 x에 남겨두면 정답을 그대로 알려주고 맞히라는 것과 같다.
#   게다가 test_csv에는 이 두 열이 아예 없어서 열 개수도 안 맞는다.
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
# 마지막 층에도 relu를 준 이유: 대여량(count)은 음수가 나올 수 없는데
# relu가 음수를 0으로 잘라주기 때문이다.


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam' )
model.fit(x_train,y_train, epochs = 2000, batch_size=200)

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

############## 대회 제출 파일 만들기 ##############
# 여기서부터는 훈련이 아니라 제출용이다.
# 정답이 없는 test_csv로 예측해서, 그 값을 sampleSubmission의 count 열에 채워 저장한다.
y_submit = model.predict(test_csv)  #test_csv를 pred 에(예측값에 넣고) y_서브밋에 저장
submission['count'] = y_submit # Y_서브밋에 저장된 내용을 서브미션 파일에 "count" 컬럼에 내용 추가 

submission.to_csv(path + 'submit/' + 'submit_0904_3.csv')


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