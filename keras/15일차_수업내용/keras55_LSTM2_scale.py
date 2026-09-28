import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

#1 데이터
x = np.array([[1,2,3,],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60],  
              ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0],x.shape[1],1)
print(x.shape) #(13, 3, 1)

#2모델구성
model = Sequential()
model.add(LSTM(units=512, input_length=3, input_dim=1))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3컴파일,훈련
# es = EarlyStopping(
#     mode='auto',
#     monitor='val_loss',
#     patience=40,
#     restore_best_weights=True,
#     verbose=1,
# )
# mcp = ModelCheckpoint(
#     mode='auto',
#     monitor='val_loss',
#     patience=40,
#     save_weights_only=True,
#     verbose=1,
# )
# lr = ReduceLROnPlateau(
#     mode='auto',
#     monitor='val_loss',
#     factor=0.5,
#     patience=20,
#     verbose=1,
    
# )

model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs=500,)
# exit()

#4.평가,예측

results = model.evaluate(x,y)

print('loss', results)

x_predict = np.array([50,60,70]).reshape(-1,3,1)
y_predict = model.predict(x_predict)

print('[50,60,70]의 결과 :', y_predict)



'''

epochs = 500 통일
LSTM units =10
[50,60,70]의 결과 : [[71.632996]]

[50,60,70]의 결과 : [[75.18008]]

LSTM units =3o
[50,60,70]의 결과 : [[77.363266]]


LSTM units =64
[50,60,70]의 결과 : [[78.1663]]

LSTM units =128
[50,60,70]의 결과 : [[79.03345]]


LSTM units =256
[50,60,70]의 결과 : [[79.46067]]

LSTM units =512
[50,60,70]의 결과 : [[79.51117]]
'''