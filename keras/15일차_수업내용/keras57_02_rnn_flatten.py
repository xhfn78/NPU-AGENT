import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU,Dropout,Flatten,SimpleRNN
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

#2.모델구성
# model = Sequential()
# model.add(LSTM(units =10, input_shape=(3,1),return_sequences=True,))
# model.add(LSTM(5,return_sequences=True,))#LSTM 다층연결시 return_sequences=True를 안넣으면 2차원값을 리턴시켜서 에러남
# model.add(LSTM(5,))
# model.add(Flatten())
# model.add(Dense(8))
# model.add(Dense(1))
# model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  lstm (LSTM)                 (None, 3, 10)             480       
                                                                 
#  lstm_1 (LSTM)               (None, 3, 5)              320       
                                                                 
#  lstm_2 (LSTM)               (None, 5)                 220       
                                                                 
#  dense (Dense)               (None, 8)                 48        
                                                                 
#  dense_1 (Dense)             (None, 1)                 9         
                                                                 
# =================================================================
# Total params: 1,077
# Trainable params: 1,077
# Non-trainable params: 0
# _______________________________________________________________

# #2모델구성
model = Sequential()
model.add(SimpleRNN(units=10, input_shape=(3,1), return_sequences=True,))
model.add(SimpleRNN(units=5, return_sequences=True, ))
# model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


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
기존모델 
loss 0.0013775908155366778
1/1 [==============================] - 0s 282ms/step
[50,60,70]의 결과 : [[79.51775]]

lstm 다층연결 
[50,60,70]의 결과 : [[[20.216608]
  [20.828974]
  [20.783655]]]
flatten적용
[50,60,70]의 결과 : [[71.39551]]
'''