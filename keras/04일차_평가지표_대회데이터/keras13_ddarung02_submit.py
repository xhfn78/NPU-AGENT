#https://dacon.io/competitions/open/235576/codeshare 대회 주소
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터

path = 'c:\\study\\_data\\ddarung\\'   #<<< \\두개써도 가능
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


#####################submit 작업 ################################
# print(test_csv.info())


######################결측치 처리 2.평균값 넣기 ####################
test_csv = test_csv.fillna(test_csv.mean())   ##
# print(test_csv.info()) #(715, 9)
# print(test_csv.shape) #(715, 9)






# exit()

#2.모델구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3.컴파일 ,훈련

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train , epochs= 500 , batch_size=32)

#4.평가 예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2결과값: ' ,r2)

y2_pred = model.predict(test_csv)

mse = mean_squared_error(y_test,y_predict)
print('mse : ', mse)

def RMSE(y_test, y_predict):  #RMSE 함수정의
    return np.sqrt(mean_squared_error(y_test,y_predict))  #np.sqrt하면 mse에 루트가 씌워짐

rmse = RMSE(y_test, y_predict)

print('RMSE : ', rmse) 



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


# [715 rows x 1 columns]
# (715, 1)

submission.to_csv(path + 'submit/' + 'submit_0904_1148.csv')



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
random : 337
train_size = 0.75
epochs = 50
batch_size = 3
결과
loss
rmse: 54.06878082204322
r2 :0.59
"""