#11_3 COPY
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

path = './_save/keras30/' 

#1.데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)


from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler  #preprocessing(전처리)
from sklearn.preprocessing import RobustScaler


##############################################################################
scaler = RobustScaler()
##############################################################################
scaler.fit(x_train) # x 값을  MinMaxScaler으로 실행시킬 준비
x_train = scaler.fit_transform(x_train) # 0~1 값 변환 사이로변환
x_test = scaler.transform(x_test) 

#2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=13))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))



#3.컴파일,훈련
import datetime
date = datetime.datetime.now() 
print(date) #2026-09-14 11:41:17.149590
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')                
print(date)
print(type(date))
path = './_save/keras30/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = ''.join([path,'k30_',date,'-',filename])


model.compile(loss='mse', optimizer= 'adam') #mse= 원값에서 예측값 뺴고 나온값을 제곱 > 다 더해서 갯수만큼 엔빵
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode= 'auto',
    patience= 15,
    restore_best_weights=True,
)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode= 'auto',
    save_best_only=True,
    filepath = filepath,
    verbose=1,
)


hist = model.fit(x_train,y_train, 
                 epochs=500, 
                 batch_size=16 ,
                 validation_split=0.2,
                 callbacks =[es,mcp],
                 )


#4.평가,예측
print("=========================================")

#4.평가 예측

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

