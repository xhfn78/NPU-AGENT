# [실습] Dropout 적용 - 캘리포니아 주택 가격 (회귀)
#
# 층 사이에 Dropout을 넣어서 과적합을 줄여본다.
# Dropout을 넣기 전(keras31_MCP_save_01)과 결과가 어떻게 달라지는지 비교해본다.
#30-1카피
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,MaxPool2D,Conv2D,GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler  #preprocessing(전처리)
from sklearn.preprocessing import RobustScaler
import numpy as np
import time

path = './_save/keras30/'  

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
\

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.75,
    random_state=333
)


x_train = x_train/255.
x_test = x_test/255.
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
##############################################################################
x_test = scaler.transform(x_test) 


x_train =x_train.reshape(-1,8,1,1)
x_test = x_test.reshape(-1,8,1,1)


print(x_train.shape,y_train.shape) #(15480, 8) (15480,)



#2. 모델구성
model = Sequential()
model.add(Conv2D(32,(2,1) ,input_shape=(8,1,1,),activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,1),activation='relu'))
model.add(Dropout(0.3))
model.add(GlobalAveragePooling2D())
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience= 20,
    verbose=1,
    restore_best_weights=True,
)
# ModelCheckpoint(MCP)란?
#   훈련 도중 val_loss가 가장 좋았던 순간의 모델을 파일로 자동 저장해주는 콜백이다.
#
#   EarlyStopping의 restore_best_weights=True 와 뭐가 다른가?
#     EarlyStopping : 최적 가중치를 "메모리 안의 model"에 되돌려준다. 프로그램이 끝나면 사라진다.
#     ModelCheckpoint: 최적 시점의 모델을 "파일"로 남긴다. 나중에 다시 불러 쓸 수 있다.
#
#   주요 옵션
#     monitor='val_loss'    → 무엇을 기준으로 좋고 나쁨을 볼지
#     save_best_only=True   → 좋아졌을 때만 덮어쓴다 (False면 매 epoch 저장해서 파일이 쏟아진다)
#     filepath              → 저장할 경로와 파일명
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode= 'auto',
    save_best_only=True,
    filepath = path + 'keras30_mcp1.keras',
    verbose=1,
)
start_time = time.time()  #현재 시간을 반환 ,시작시간
hist = model.fit(x_train,y_train, 
                 epochs=500, 
                 batch_size=32,
                 validation_split=0.2, 
                 callbacks =[es,mcp] ,
                 verbose=1,
                 )
end_time = time.time()  #훈련 끝난 시간을 반환 , 끝시간



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



'''
# r2결과값:  0.7259173310645768
# mse :  0.350731508708081
# RMSE :  0.592225893311058


# val_loss: 0.5916
# Epoch 97: early stopping
# 162/162 [==============================] - 0s 1ms/step - loss: 0.5469
# loss: 0.5468983054161072
# 162/162 [==============================] - 0s 1ms/step
# r2 :  0.5726208249558167
# mse :  0.5468982896141396
# RMSE :  0.7395257193729908

'''
