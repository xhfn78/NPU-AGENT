import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau



a = np.array(range(1,101))
x_predict = np.array(range(96,106))

size = 6   #timestep 사이즈 

print(a.shape) # (10, 2)


def split_x(dataset,size):
    aaa = []
    for i in range(len(dataset)- size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)


bbb = split_x (a , size)
print(bbb.shape)  #(95, 6)

x = bbb[ :, :-1]  #  :, : -1,:
y = bbb[ :,-1]
# y = bbb[ :,-1,-1]
# exit()
# x = bbb[:,:-1]
# print(x,y)
# print(x.shape,y.shape)
# exit()
#2모델구성
model = Sequential()
model.add(LSTM(units=15, input_shape=(5, 2), return_sequences=True))
model.add(LSTM(units=32,))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3.컴파일훈련

model.compile(loss = 'mse',optimizer = 'adam')
model.fit(x,y,epochs=500,)
# exit()

#4.평가,예측

results = model.evaluate(x,y)

print('loss', results)

current = np.array([96,97,98,99,100,101,102,103,104,105], dtype=np.float32)

predictions = []

for i in range(6):

    x_predict = current.reshape(1,5,2)

    pred = model.predict(x_predict, verbose=0)[0][0]

    predictions.append(pred)

    current = np.append(current[1:], pred)

print("101~106 예측 :", predictions)
#로스는 0.1이하



'''
101~106 예측 : [100.77253, 101.51935, 102.19928, 102.8242, 103.36659, 103.86438]
'''